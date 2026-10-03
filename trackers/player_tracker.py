import os
import cv2
import pickle
import numpy as np
import pandas as pd
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    """
    Tennis Player Tracker supporting TensorFlow models and YOLO26 tracking.
    """
    def __init__(self, model_path='yolo26s.pt', load_model=True, device=None):
        self.model_path = model_path
        self.device = device
        self.backend = None
        self.model = None
        self.racket_track_counts = {}

        if load_model:
            self._load_model(model_path)

    def _load_model(self, model_path):
        if not model_path:
            model_path = 'yolo26s.pt'
        
        # Check if file exists in models/ folder
        if not os.path.exists(str(model_path)):
            candidate = os.path.join('models', os.path.basename(str(model_path)))
            if os.path.exists(candidate):
                model_path = candidate

        # Check if model_path is a TensorFlow SavedModel
        if os.path.isdir(str(model_path)) and (
            os.path.exists(os.path.join(model_path, "saved_model.pb")) or
            os.path.exists(os.path.join(model_path, "fingerprint.pb"))
        ):
            import tensorflow as tf
            self.tf = tf
            self.backend = 'tf_saved_model'
            self.tf_model = tf.saved_model.load(model_path)
            self.infer_fn = self.tf_model.signatures["serving_default"]
            print(f"[PlayerTracker] Loaded TensorFlow SavedModel: {model_path}")
        else:
            try:
                from ultralytics import YOLO
                self.backend = 'ultralytics'
                self.model = YOLO(model_path)
                print(f"[PlayerTracker] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[PlayerTracker] Fallback loading model {model_path}: {e}")

    def choose_and_filter_players(self, court_keypoints, player_detections, match_mode="auto"):
        """
        Robustly choose active tennis players across video frames:
        - Singles mode (2 players): Player 1 (Near court) vs Player 2 (Far court).
        - Doubles mode (4 players): Team 1 (P1 & P3 Near court) vs Team 2 (P2 & P4 Far court).
        - Auto mode: Dynamically detects whether court has 2 players (Singles) or 4 players (Doubles).
        Filters out all spectators, chair umpires, line judges, and ball boys using
        perspective-aware trapezoid court perimeter gating and cumulative kinetic activity scoring.
        """
        if not player_detections or len(player_detections) == 0:
            return player_detections

        # Determine net line and perspective court trapezoid dynamically from court keypoints
        ref_kps = court_keypoints[0] if (court_keypoints is not None and len(court_keypoints) > 0 and hasattr(court_keypoints[0], '__len__')) else court_keypoints
        has_court = (ref_kps is not None and len(ref_kps) >= 28)

        if has_court:
            x0, y0 = float(ref_kps[0]), float(ref_kps[1])  # Far-left baseline corner
            x1, y1 = float(ref_kps[2]), float(ref_kps[3])  # Far-right baseline corner
            x2, y2 = float(ref_kps[4]), float(ref_kps[5])  # Near-left baseline corner
            x3, y3 = float(ref_kps[6]), float(ref_kps[7])  # Near-right baseline corner

            y_far = (y0 + y1) / 2.0
            y_near = (y2 + y3) / 2.0
            court_h = max(1.0, y_near - y_far)

            # Net line Y is the midpoint between top and bottom service lines (indices 8, 10)
            net_y = (float(ref_kps[2 * 8 + 1]) + float(ref_kps[2 * 10 + 1])) / 2.0

            # Vertical depth bounds: players can run behind far baseline or near baseline
            # Far baseline margin: 0.16 * court_h (rejects spectators/linesmen standing at back wall)
            # Near baseline margin: 0.35 * court_h (players running wide towards broadcast camera)
            y_min = y_far - 0.16 * court_h
            y_max = y_near + 0.35 * court_h

            def is_inside_playing_perimeter(cx, cy):
                if cy < y_min or cy > y_max:
                    return False
                # Perspective interpolation factor along court depth
                alpha = np.clip((cy - y_far) / court_h, -0.16, 1.35)
                x_left = x0 + alpha * (x2 - x0)
                x_right = x1 + alpha * (x3 - x1)
                court_w = max(10.0, x_right - x_left)
                # Players run outside doubles sidelines by at most 22% of local court width
                margin = 0.22 * court_w
                return (x_left - margin) <= cx <= (x_right + margin)
        else:
            net_y = 530.0
            def is_inside_playing_perimeter(cx, cy):
                return (100.0 <= cx <= 1820.0 and 80.0 <= cy <= 1040.0)

        # Collect track statistics across all frames for tracks inside the playing perimeter
        track_stats = {}
        for p_dict in player_detections:
            for track_id, bbox in p_dict.items():
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                # Strictly reject spectators, chair umpires, ball boys outside playing trapezoid
                if not is_inside_playing_perimeter(cx, cy):
                    continue

                if track_id not in track_stats:
                    track_stats[track_id] = {'ys': [], 'xs': []}
                track_stats[track_id]['ys'].append(cy)
                track_stats[track_id]['xs'].append(cx)

        # Classify candidate tracks for Near court (y > net_y) and Far court (y <= net_y)
        # Active tennis players move dynamically across court; stationary ball boys / judges are heavily penalized
        near_candidates = {}
        far_candidates = {}
        total_frames = len(player_detections)

        for track_id, stats in track_stats.items():
            count = len(stats['ys'])
            if count < 5:
                continue

            pts = np.column_stack((stats['xs'], stats['ys']))
            diffs = np.diff(pts, axis=0)
            path_len = float(np.sum(np.hypot(diffs[:, 0], diffs[:, 1]))) if len(pts) > 1 else 0.0
            avg_y = float(np.mean(stats['ys']))
            dx = float(max(stats['xs']) - min(stats['xs']))
            dy = float(max(stats['ys']) - min(stats['ys']))

            # Real tennis player moves across court; stationary noise is heavily penalized
            mobility = (dx / 300.0) * (path_len / 500.0)
            activity_mult = (1.0 + min(mobility, 10.0))
            if dx < 40.0 and dy < 25.0:
                activity_mult *= 0.10  # Heavy penalty for stationary judge / ball boy

            # Semantic racket carrier bonus: players holding tennis rackets receive massive boost
            racket_hits = getattr(self, 'racket_track_counts', {}).get(track_id, 0)
            if racket_hits >= 2:
                activity_mult *= 2.5

            score = count * activity_mult

            if avg_y > net_y:
                near_candidates[track_id] = score
            else:
                far_candidates[track_id] = score

        # Fallback if no candidate found: pick track with highest frames on that side
        if not near_candidates:
            near_by_count = {t: len(s['ys']) for t, s in track_stats.items() if np.mean(s['ys']) > net_y}
            if near_by_count:
                top_near = max(near_by_count.keys(), key=lambda t: near_by_count[t])
                near_candidates[top_near] = float(near_by_count[top_near])

        if not far_candidates:
            far_by_count = {t: len(s['ys']) for t, s in track_stats.items() if np.mean(s['ys']) <= net_y}
            if far_by_count:
                top_far = max(far_by_count.keys(), key=lambda t: far_by_count[t])
                far_candidates[top_far] = float(far_by_count[top_far])

        # Sort candidates by overall kinetic activity score
        sorted_near = sorted(near_candidates.keys(), key=lambda t: near_candidates[t], reverse=True)
        sorted_far = sorted(far_candidates.keys(), key=lambda t: far_candidates[t], reverse=True)

        if not sorted_near:
            sorted_near = [1]
            near_candidates[1] = 1.0
        if not sorted_far:
            sorted_far = [2]
            far_candidates[2] = 1.0

        # Determine Singles vs Doubles match
        is_doubles = False
        if match_mode == "doubles":
            is_doubles = True
        elif match_mode == "auto":
            # Auto-detect Doubles ONLY if both sides have 2 distinct active players
            # with high frame presence (>= 45%), significant path travel, and lateral separation
            min_presence = max(15, int(total_frames * 0.45))
            strong_near = [t for t in sorted_near if len(track_stats[t]['xs']) >= min_presence]
            strong_far = [t for t in sorted_far if len(track_stats[t]['xs']) >= min_presence]
            if len(strong_near) >= 2 and len(strong_far) >= 2:
                near_sep = abs(np.mean(track_stats[strong_near[0]]['xs']) - np.mean(track_stats[strong_near[1]]['xs']))
                far_sep = abs(np.mean(track_stats[strong_far[0]]['xs']) - np.mean(track_stats[strong_far[1]]['xs']))
                if near_sep >= 120.0 and far_sep >= 100.0:
                    is_doubles = True

        if is_doubles:
            primary_p1 = sorted_near[0]
            secondary_p3 = sorted_near[1] if len(sorted_near) > 1 else sorted_near[0]
            primary_p2 = sorted_far[0]
            secondary_p4 = sorted_far[1] if len(sorted_far) > 1 else sorted_far[0]
            print(f"[PlayerTracker] Mode: DOUBLES (4 Players)")
            print(f"  - Team 1 (Near): P1 (Track {primary_p1}), P3 (Track {secondary_p3}) | Candidates: {sorted_near}")
            print(f"  - Team 2 (Far) : P2 (Track {primary_p2}), P4 (Track {secondary_p4}) | Candidates: {sorted_far}")
        else:
            primary_p1 = sorted_near[0]
            primary_p2 = sorted_far[0]
            secondary_p3 = None
            secondary_p4 = None
            print(f"[PlayerTracker] Mode: SINGLES (2 Players)")
            print(f"  - Near Player (P1): Primary Track {primary_p1}, Candidates: {sorted_near}")
            print(f"  - Far Player (P2) : Primary Track {primary_p2}, Candidates: {sorted_far}")

        # Frame-by-frame player assignment with temporal continuity & track stitching
        filtered_player_detections = []
        last_positions = {}

        for player_dict in player_detections:
            frame_res = {}

            # Filter out any detection in current frame outside playing perimeter
            valid_p_dict = {}
            for t, box in player_dict.items():
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                if is_inside_playing_perimeter(cx, cy):
                    valid_p_dict[t] = box

            # --- 1. Near Court Assignment (Team 1) ---
            if not is_doubles:
                # Singles: P1
                if primary_p1 in valid_p_dict:
                    frame_res[1] = valid_p_dict[primary_p1]
                else:
                    p1_cands = [t for t in sorted_near if t in valid_p_dict]
                    if p1_cands:
                        best_p1 = max(p1_cands, key=lambda t: near_candidates[t])
                        frame_res[1] = valid_p_dict[best_p1]
                if 1 in frame_res:
                    last_positions[1] = ((frame_res[1][0] + frame_res[1][2]) / 2.0, (frame_res[1][1] + frame_res[1][3]) / 2.0)
            else:
                # Doubles: P1 and P3 on Near Court
                available_near = [t for t in valid_p_dict if t in near_candidates]
                used_tracks = set()
                if primary_p1 in available_near:
                    frame_res[1] = valid_p_dict[primary_p1]
                    used_tracks.add(primary_p1)
                if secondary_p3 in available_near and secondary_p3 != primary_p1:
                    frame_res[3] = valid_p_dict[secondary_p3]
                    used_tracks.add(secondary_p3)

                remaining = [t for t in available_near if t not in used_tracks]
                for slot in (1, 3):
                    if slot not in frame_res and remaining:
                        if slot in last_positions:
                            best_t = min(remaining, key=lambda t: np.hypot(
                                (valid_p_dict[t][0] + valid_p_dict[t][2]) / 2.0 - last_positions[slot][0],
                                (valid_p_dict[t][1] + valid_p_dict[t][3]) / 2.0 - last_positions[slot][1]
                            ))
                        else:
                            best_t = remaining[0]
                        frame_res[slot] = valid_p_dict[best_t]
                        remaining.remove(best_t)

                for slot in (1, 3):
                    if slot in frame_res:
                        last_positions[slot] = ((frame_res[slot][0] + frame_res[slot][2]) / 2.0, (frame_res[slot][1] + frame_res[slot][3]) / 2.0)

            # --- 2. Far Court Assignment (Team 2) ---
            if not is_doubles:
                # Singles: P2
                if primary_p2 in valid_p_dict:
                    frame_res[2] = valid_p_dict[primary_p2]
                else:
                    p2_cands = [t for t in sorted_far if t in valid_p_dict]
                    if p2_cands:
                        best_p2 = max(p2_cands, key=lambda t: far_candidates[t])
                        frame_res[2] = valid_p_dict[best_p2]
                if 2 in frame_res:
                    last_positions[2] = ((frame_res[2][0] + frame_res[2][2]) / 2.0, (frame_res[2][1] + frame_res[2][3]) / 2.0)
            else:
                # Doubles: P2 and P4 on Far Court
                available_far = [t for t in valid_p_dict if t in far_candidates]
                used_tracks = set()
                if primary_p2 in available_far:
                    frame_res[2] = valid_p_dict[primary_p2]
                    used_tracks.add(primary_p2)
                if secondary_p4 in available_far and secondary_p4 != primary_p2:
                    frame_res[4] = valid_p_dict[secondary_p4]
                    used_tracks.add(secondary_p4)

                remaining = [t for t in available_far if t not in used_tracks]
                for slot in (2, 4):
                    if slot not in frame_res and remaining:
                        if slot in last_positions:
                            best_t = min(remaining, key=lambda t: np.hypot(
                                (valid_p_dict[t][0] + valid_p_dict[t][2]) / 2.0 - last_positions[slot][0],
                                (valid_p_dict[t][1] + valid_p_dict[t][3]) / 2.0 - last_positions[slot][1]
                            ))
                        else:
                            best_t = remaining[0]
                        frame_res[slot] = valid_p_dict[best_t]
                        remaining.remove(best_t)

                for slot in (2, 4):
                    if slot in frame_res:
                        last_positions[slot] = ((frame_res[slot][0] + frame_res[slot][2]) / 2.0, (frame_res[slot][1] + frame_res[slot][3]) / 2.0)

            filtered_player_detections.append(frame_res)

        return filtered_player_detections

    def interpolate_player_positions(self, player_detections):
        """
        Interpolate missing player detections using linear interpolation and backfill/forward-fill,
        ensuring all active players (Singles: P1, P2 | Doubles: P1, P2, P3, P4) are present in every frame.
        """
        if not player_detections:
            return player_detections

        # Dynamically determine all active player IDs in the match
        all_pids = sorted(list(set(pid for f in player_detections for pid in f.keys())))
        if not all_pids:
            return player_detections

        interpolated_dfs = {}
        for pid in all_pids:
            p_list = [f.get(pid, []) for f in player_detections]
            df_p = pd.DataFrame([x if len(x) == 4 else [np.nan] * 4 for x in p_list], columns=['x1', 'y1', 'x2', 'y2'])
            df_p = df_p.interpolate().bfill().ffill()
            interpolated_dfs[pid] = df_p.to_numpy().tolist()

        interpolated_detections = []
        for i in range(len(player_detections)):
            d = {}
            for pid in all_pids:
                box = interpolated_dfs[pid][i]
                if not np.isnan(box[0]):
                    d[pid] = box
            interpolated_detections.append(d)

        return interpolated_detections

    def choose_players(self, court_keypoints, player_dict):
        """
        Select the 2 tracked persons with the smallest distance to court keypoints.
        """
        distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_center_of_bbox(bbox)

            min_distance = float('inf')
            for i in range(0, len(court_keypoints), 2):
                court_keypoint = (court_keypoints[i], court_keypoints[i + 1])
                distance = measure_distance(player_center, court_keypoint)
                if distance < min_distance:
                    min_distance = distance
            distances.append((track_id, min_distance))

        # Sort the distances in ascending order
        distances.sort(key=lambda x: x[1])

        # Choose up to the first 2 tracks
        chosen_players = [d[0] for d in distances[:2]]
        return chosen_players

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Detect players across all video frames, optionally reading from stub cache.
        """
        player_detections = []

        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                player_detections = pickle.load(f)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(player_detections, f)

        return player_detections

    def detect_frame(self, frame):
        """
        Detect and track players (person class) in a single frame.
        Returns a dict: {track_id: [x1, y1, x2, y2]}
        """
        player_dict = {}

        if self.backend == 'ultralytics' and self.model is not None:
            track_kwargs = {'persist': True, 'verbose': False, 'classes': [0, 38]}
            if self.device is not None:
                track_kwargs['device'] = self.device
            results = self.model.track(frame, **track_kwargs)[0]
            id_name_dict = results.names

            persons = []
            rackets = []
            for box in results.boxes:
                cls_id = int(box.cls.tolist()[0])
                cls_name = id_name_dict.get(cls_id, "")
                result = box.xyxy.tolist()[0]
                if cls_name == "person":
                    track_id = int(box.id.tolist()[0]) if box.id is not None else 0
                    player_dict[track_id] = result
                    persons.append((track_id, result))
                elif cls_name == "tennis racket":
                    rackets.append(result)

            # Associate detected tennis rackets with person bounding boxes
            for r in rackets:
                rcx = (r[0] + r[2]) / 2.0
                rcy = (r[1] + r[3]) / 2.0
                for tid, pbox in persons:
                    # Check if racket center is inside person box extended by 35px
                    if (pbox[0] - 35 <= rcx <= pbox[2] + 35) and (pbox[1] - 35 <= rcy <= pbox[3] + 35):
                        self.racket_track_counts[tid] = self.racket_track_counts.get(tid, 0) + 1

        elif self.backend == 'tf_saved_model':
            # Preprocess frame for TensorFlow model
            h, w = frame.shape[:2]
            img_resized = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_tensor = self.tf.convert_to_tensor(img_rgb[np.newaxis, ...], dtype=self.tf.float32) / 255.0

            outputs = self.infer_fn(input_tensor)
            output_tensor = list(outputs.values())[0].numpy()
            output = np.squeeze(output_tensor)
            if output.shape[0] < output.shape[1]:
                output = output.T

            track_idx = 1
            for det in output:
                score = float(det[4])
                if score >= 0.3:
                    xc, yc, bw, bh = det[0], det[1], det[2], det[3]
                    x1 = (xc - bw / 2) * (w / 640.0)
                    y1 = (yc - bh / 2) * (h / 640.0)
                    x2 = (xc + bw / 2) * (w / 640.0)
                    y2 = (yc + bh / 2) * (h / 640.0)
                    player_dict[track_idx] = [x1, y1, x2, y2]
                    track_idx += 1

        return player_dict

    def draw_player_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw a stylish broadcast perspective ellipse under the player's feet with an ID badge.
        """
        x1, y1, x2, y2 = bbox
        x_center = int((x1 + x2) / 2)
        width = max(10, x2 - x1)
        
        # 1. Perspective ellipse arc on the court ground around feet
        axes_w = max(14, int(width * 0.70))
        axes_h = max(5, int(0.35 * axes_w))
        cv2.ellipse(
            frame,
            center=(x_center, int(y2)),
            axes=(axes_w, axes_h),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=3,
            lineType=cv2.LINE_AA
        )
        
        # 2. Sleek filled badge for Player ID
        if track_id is not None:
            rect_w = max(36, min(50, int(axes_w * 0.88)))
            rect_h = max(16, min(22, int(rect_w * 0.48)))
            rx1 = int(x_center - rect_w // 2)
            ry1 = int(y2 + 4)
            rx2 = int(x_center + rect_w // 2)
            ry2 = int(y2 + 4 + rect_h)
            
            # Subtle shadow & colored badge with white border
            cv2.rectangle(frame, (rx1 + 1, ry1 + 1), (rx2 + 1, ry2 + 1), (0, 0, 0), -1)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), color, -1)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)
            
            text = f"P{track_id}"
            text_color = (0, 0, 0) if color in [(0, 255, 255), (0, 140, 255)] else (255, 255, 255)
            font_scale = 0.45 if rect_w < 42 else 0.5
            cv2.putText(
                frame,
                text,
                (rx1 + 6, ry1 + rect_h - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                text_color,
                2,
                cv2.LINE_AA
            )

    def draw_bboxes(self, video_frames, player_detections, draw_mode="ellipse"):
        """
        Draw player annotations on video frames.
        Supports both Singles (P1 vs P2) and Doubles (P1 & P3 vs P2 & P4).
        """
        player_colors = {
            1: (0, 0, 255),      # Red for P1 (Team 1 - Near Court)
            3: (0, 140, 255),    # Amber/Orange for P3 (Team 1 - Near Court)
            2: (0, 255, 255),    # Yellow for P2 (Team 2 - Far Court)
            4: (255, 0, 200),    # Magenta/Purple for P4 (Team 2 - Far Court)
        }

        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                color = player_colors.get(track_id, (0, 255, 0))

                if draw_mode == "ellipse":
                    self.draw_player_ellipse(frame, bbox, color, track_id=track_id)
                else:
                    cv2.putText(
                        frame,
                        f"Player P{track_id}",
                        (int(x1), int(max(15, y1 - 10))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        color,
                        2
                    )
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

            output_video_frames.append(frame)

        return output_video_frames
