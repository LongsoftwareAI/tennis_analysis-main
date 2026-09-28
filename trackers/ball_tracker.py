import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf

class BallTracker:
    """
    Tennis Ball Tracker supporting pure TensorFlow (SavedModel, TFLite) 
    as well as YOLO26 models.
    """
    def __init__(self, model_path):
        self.model_path = model_path
        self.backend = None  # 'tf_saved_model', 'tflite', or 'ultralytics'
        self.model = None

        self._load_model(model_path)

    def _load_model(self, model_path):
        if not model_path:
            raise ValueError("Model path must be specified.")

        # Check if file exists in models/ folder
        if not os.path.exists(str(model_path)):
            candidate = os.path.join('models', os.path.basename(str(model_path)))
            if os.path.exists(candidate):
                model_path = candidate

        # Check if model_path is a TensorFlow TFLite file
        if str(model_path).endswith('.tflite') and os.path.exists(model_path):
            self.backend = 'tflite'
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[BallTracker] Loaded TensorFlow Lite model: {model_path}")

        # Check if model_path is a TensorFlow SavedModel directory
        elif os.path.isdir(model_path) and (
            os.path.exists(os.path.join(model_path, "saved_model.pb")) or 
            os.path.exists(os.path.join(model_path, "fingerprint.pb"))
        ):
            self.backend = 'tf_saved_model'
            self.tf_model = tf.saved_model.load(model_path)
            self.infer_fn = self.tf_model.signatures["serving_default"]
            print(f"[BallTracker] Loaded TensorFlow SavedModel from: {model_path}")

        # Otherwise load with YOLO (e.g., YOLO26: yolo26l.pt)
        else:
            try:
                from ultralytics import YOLO
                self.backend = 'ultralytics'
                self.model = YOLO(model_path)
                print(f"[BallTracker] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[BallTracker] Fallback loading model {model_path}: {e}")

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolate missing ball detections using velocity-gated outlier rejection
        followed by linear interpolation and light rolling smoothing.
        """
        ball_positions_list = [x.get(1, []) if (1 in x and len(x[1]) == 4) else [np.nan, np.nan, np.nan, np.nan] for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        # Outlier spike rejection before interpolation:
        # Detect sudden teleportation spikes (e.g. jumping > 85 px in 1 frame and returning)
        mid_x = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2.0
        mid_y = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2.0

        for i in range(1, len(df_ball_positions) - 1):
            if pd.notna(mid_x.iloc[i]):
                prev_valid = mid_x.iloc[:i].last_valid_index()
                next_valid = mid_x.iloc[i+1:].first_valid_index()
                if prev_valid is not None and next_valid is not None:
                    dt_prev = i - prev_valid
                    dt_next = next_valid - i
                    dist_prev = np.sqrt((mid_x.iloc[i] - mid_x.iloc[prev_valid])**2 + (mid_y.iloc[i] - mid_y.iloc[prev_valid])**2)
                    dist_next = np.sqrt((mid_x.iloc[next_valid] - mid_x.iloc[i])**2 + (mid_y.iloc[next_valid] - mid_y.iloc[i])**2)
                    dist_span = np.sqrt((mid_x.iloc[next_valid] - mid_x.iloc[prev_valid])**2 + (mid_y.iloc[next_valid] - mid_y.iloc[prev_valid])**2)
                    
                    # If current point spikes away from both neighbors while neighbors are close together
                    if dist_prev > 80.0 * dt_prev and dist_next > 80.0 * dt_next and dist_span < 120.0 * (next_valid - prev_valid):
                        df_ball_positions.iloc[i] = [np.nan, np.nan, np.nan, np.nan]

        # Interpolate missing frames
        df_ball_positions = df_ball_positions.interpolate(method='linear')
        df_ball_positions = df_ball_positions.bfill().ffill()

        # Normalize bounding box around center (cx, cy) to keep box tightly locked to ball
        # without bounding box aspect-ratio flickering
        mid_x = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2.0
        mid_y = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2.0
        box_w = (df_ball_positions['x2'] - df_ball_positions['x1']).median()
        box_h = (df_ball_positions['y2'] - df_ball_positions['y1']).median()
        half_w = max(7.0, min(15.0, box_w / 2.0 if pd.notna(box_w) else 9.0))
        half_h = max(7.0, min(15.0, box_h / 2.0 if pd.notna(box_h) else 9.0))

        df_ball_positions['x1'] = mid_x - half_w
        df_ball_positions['y1'] = mid_y - half_h
        df_ball_positions['x2'] = mid_x + half_w
        df_ball_positions['y2'] = mid_y + half_h

        interpolated_positions = [{1: x} for x in df_ball_positions.to_numpy().tolist()]
        return interpolated_positions

    def get_ball_shot_frames(self, ball_positions, player_positions=None):
        """
        Detect frames where a shot occurred based on vertical trajectory inflection.
        Also detects unclosed final return shots near the end of a rally using player proximity.
        """
        ball_positions_list = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])

        df_ball_positions['ball_hit'] = 0
        df_ball_positions['mid_y'] = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2
        df_ball_positions['mid_x'] = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2
        df_ball_positions['mid_y_rolling_mean'] = df_ball_positions['mid_y'].rolling(
            window=5, min_periods=1, center=False
        ).mean()
        df_ball_positions['delta_y'] = df_ball_positions['mid_y_rolling_mean'].diff()

        minimum_change_frames_for_hit = 25
        limit = len(df_ball_positions) - int(minimum_change_frames_for_hit * 1.2)
        for i in range(1, max(1, limit)):
            negative_position_change = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[i + 1] < 0
            positive_position_change = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[i + 1] > 0

            if negative_position_change or positive_position_change:
                bx = df_ball_positions['mid_x'].iloc[i]
                by = df_ball_positions['mid_y'].iloc[i]

                # 1. Height filter: reject airborne turning points in the sky (e.g. lob apex at Y < 120px)
                if not np.isnan(by) and by < 120.0:
                    continue

                # 2. Player proximity check: a true shot must occur within reach of a player's racket
                if player_positions is not None and not np.isnan(bx) and not np.isnan(by):
                    min_p_dist = float('inf')
                    for check_f in range(max(0, i - 4), min(len(player_positions), i + 5)):
                        p_dict = player_positions[check_f]
                        for p_id in (1, 2):
                            if p_id in p_dict and len(p_dict[p_id]) == 4:
                                p_bbox = p_dict[p_id]
                                p_center = ((p_bbox[0] + p_bbox[2]) / 2.0, (p_bbox[1] + p_bbox[3]) / 2.0)
                                dist = np.hypot(bx - p_center[0], by - p_center[1])
                                if dist < min_p_dist:
                                    min_p_dist = dist
                    # If ball is far away from both players, it is an airborne apex or flight turning point, not a shot
                    if min_p_dist > 250.0:
                        continue

                change_count = 0
                max_check = min(len(df_ball_positions), i + int(minimum_change_frames_for_hit * 1.2) + 1)
                for change_frame in range(i + 1, max_check):
                    neg_following = df_ball_positions['delta_y'].iloc[i] > 0 and df_ball_positions['delta_y'].iloc[change_frame] < 0
                    pos_following = df_ball_positions['delta_y'].iloc[i] < 0 and df_ball_positions['delta_y'].iloc[change_frame] > 0

                    if negative_position_change and neg_following:
                        change_count += 1
                    elif positive_position_change and pos_following:
                        change_count += 1

                if change_count > minimum_change_frames_for_hit - 1:
                    df_ball_positions.loc[i, 'ball_hit'] = 1

        frame_nums_with_ball_hits = df_ball_positions[df_ball_positions['ball_hit'] == 1].index.tolist()

        # Filter out closely-spaced candidates (e.g. court bounce followed immediately by racket strike)
        # In tennis, consecutive shots by players cannot occur within < 22 frames (~0.8s).
        # On a groundstroke, the ball bounces first, then the player strikes it ~10-18 frames later.
        raw_shots = sorted(list(set(frame_nums_with_ball_hits)))
        filtered_shots = []
        for s in raw_shots:
            if not filtered_shots:
                filtered_shots.append(s)
            elif s - filtered_shots[-1] < 22:
                # Replace earlier bounce with the actual strike
                filtered_shots[-1] = s
            else:
                filtered_shots.append(s)

        return filtered_shots

    def detect_frames(self, frames, read_from_stub=False, stub_path=None):
        """
        Run robust ball detection and tracking on a list of video frames,
        with static background noise suppression and directional momentum vector gating.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        print(f"[BallTracker] Running candidate extraction on {len(frames)} frames...")
        # 1. Extract raw candidate detections for each frame
        raw_candidates_per_frame = []
        for frame in frames:
            cands = self._detect_candidates(frame, conf=0.10)
            raw_candidates_per_frame.append(cands)

        # 2. Automatically identify and suppress stationary background noise
        # A true static spot persists across a wide frame span (span >= 40 frames) with >= 10 detections
        # and very tight spatial radius (< 12px).
        clusters = []
        for f_idx, cands in enumerate(raw_candidates_per_frame):
            for c in cands:
                pt = np.array(c['center'])
                matched = False
                for cl in clusters:
                    if np.linalg.norm(pt - cl['center']) < 12:
                        cl['frames'].append(f_idx)
                        matched = True
                        break
                if not matched:
                    clusters.append({'center': pt, 'frames': [f_idx]})

        static_spots = []
        for cl in clusters:
            span = max(cl['frames']) - min(cl['frames'])
            if len(cl['frames']) >= 10 and span >= 40:
                static_spots.append(cl['center'])

        if static_spots:
            print(f"[BallTracker] Identified and suppressed {len(static_spots)} true static background noise spots.")

        dynamic_candidates = []
        for f_idx, cands in enumerate(raw_candidates_per_frame):
            valid = []
            for c in cands:
                if not any(np.linalg.norm(np.array(c['center']) - s) < 15 for s in static_spots):
                    valid.append(c)
            dynamic_candidates.append(valid)

        # 3. Trajectory-Consistent Tracking with Directional Momentum & Velocity Gating
        tracked_boxes = [None] * len(frames)
        current_track = None
        current_vel = None

        for f_idx, cands in enumerate(dynamic_candidates):
            if not cands:
                continue

            if current_track is None:
                high_conf = [c for c in cands if c['conf'] > 0.35]
                if high_conf:
                    best = max(high_conf, key=lambda x: x['conf'])
                    tracked_boxes[f_idx] = best['box']
                    current_track = (f_idx, np.array(best['center']))
                    current_vel = np.array([0.0, 0.0])
            else:
                last_f, last_pt = current_track
                dt = f_idx - last_f
                max_dist = max(40.0, 80.0 * dt)

                scored_cands = []
                for c in cands:
                    c_pt = np.array(c['center'])
                    disp = c_pt - last_pt
                    dist = np.linalg.norm(disp)
                    if dist > max_dist:
                        continue

                    # Directional momentum vector alignment check
                    v_norm = np.linalg.norm(current_vel) if current_vel is not None else 0.0
                    if v_norm > 10.0 and dist > 15.0:
                        cos_sim = np.dot(disp, current_vel) / (dist * v_norm)
                        # Reject backward reversals jumping into player body/hand (cos_sim < -0.2)
                        if cos_sim < -0.2 and dist > 35.0:
                            continue
                        score = c['conf'] + 0.35 * cos_sim - 0.2 * (dist / max_dist)
                    else:
                        score = c['conf'] - 0.2 * (dist / max_dist)

                    scored_cands.append((score, c))

                if scored_cands:
                    scored_cands.sort(key=lambda x: x[0], reverse=True)
                    best = scored_cands[0][1]
                    best_pt = np.array(best['center'])
                    new_vel = (best_pt - last_pt) / dt
                    current_vel = 0.7 * new_vel + 0.3 * current_vel if current_vel is not None else new_vel
                    current_track = (f_idx, best_pt)
                    tracked_boxes[f_idx] = best['box']
                elif dt > 8:
                    # Re-seed only if very strong detection
                    strong = [c for c in cands if c['conf'] > 0.50]
                    if strong:
                        best = max(strong, key=lambda x: x['conf'])
                        best_pt = np.array(best['center'])
                        current_track = (f_idx, best_pt)
                        current_vel = np.array([0.0, 0.0])
                        tracked_boxes[f_idx] = best['box']

        # Backward pass: recover earlier frames if track started late
        for f_idx in range(len(frames) - 2, -1, -1):
            if tracked_boxes[f_idx] is None and tracked_boxes[f_idx + 1] is not None:
                next_box = tracked_boxes[f_idx + 1]
                next_center = np.array([(next_box[0] + next_box[2]) / 2, (next_box[1] + next_box[3]) / 2])
                valid_cands = [c for c in dynamic_candidates[f_idx] if np.linalg.norm(np.array(c['center']) - next_center) <= 80.0]
                if valid_cands:
                    best = max(valid_cands, key=lambda x: x['conf'])
                    tracked_boxes[f_idx] = best['box']

        ball_detections = [{1: b} if b is not None else {} for b in tracked_boxes]

        if stub_path is not None:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
            print(f"[BallTracker] Saved directional momentum filtered detections to stub: {stub_path}")

        return ball_detections


    def _detect_candidates(self, frame, conf=0.10):
        """
        Extract all ball candidate boxes and confidences in a single frame.
        """
        candidates = []
        if self.backend == 'ultralytics' and self.model is not None:
            results = self.model.predict(frame, conf=conf, verbose=False)[0]
            for box in results.boxes:
                coords = box.xyxy.tolist()[0]
                score = float(box.conf[0])
                cx = (coords[0] + coords[2]) / 2.0
                cy = (coords[1] + coords[3]) / 2.0
                candidates.append({'box': coords, 'conf': score, 'center': np.array([cx, cy])})
        elif self.backend in ('tf_saved_model', 'tflite'):
            h, w = frame.shape[:2]
            img_resized = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            if self.backend == 'tf_saved_model':
                input_tensor = tf.convert_to_tensor(img_rgb[np.newaxis, ...], dtype=tf.float32) / 255.0
                outputs = self.infer_fn(input_tensor)
                output_tensor = list(outputs.values())[0].numpy()
            else:
                input_data = np.expand_dims(img_rgb.astype(np.float32) / 255.0, axis=0)
                self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
                self.interpreter.invoke()
                output_tensor = self.interpreter.get_tensor(self.output_details[0]['index'])
            boxes, scores = self._postprocess_tf_output(output_tensor, (h, w), conf_thresh=conf)
            for b, s in zip(boxes, scores):
                cx = (b[0] + b[2]) / 2.0
                cy = (b[1] + b[3]) / 2.0
                candidates.append({'box': b.tolist(), 'conf': float(s), 'center': np.array([cx, cy])})
        return candidates

    def detect_frame(self, frame, conf=0.15):
        """
        Detect tennis ball in a single frame (fallback for single-frame calls).
        Returns a dict: {1: [x1, y1, x2, y2]}
        """
        ball_dict = {}
        cands = self._detect_candidates(frame, conf=conf)
        if cands:
            best = max(cands, key=lambda x: x['conf'])
            ball_dict[1] = best['box']
        return ball_dict


    def _postprocess_tf_output(self, output_tensor, orig_shape, conf_thresh=0.15):
        """
        Parse raw YOLO tensor output from TensorFlow export into boxes and confidence scores.
        """
        orig_h, orig_w = orig_shape
        output = np.squeeze(output_tensor)  # Expected shape: (channels, num_boxes) or (num_boxes, channels)

        if output.shape[0] < output.shape[1]:
            output = output.T  # Transpose to (num_boxes, channels)

        # Standard YOLO layout: [x_center, y_center, width, height, class_scores...]
        boxes = []
        scores = []
        for det in output:
            score = float(np.max(det[4:])) if len(det) > 4 else float(det[4])
            if score >= conf_thresh:
                xc, yc, w, h = det[0], det[1], det[2], det[3]
                x1 = (xc - w / 2) * (orig_w / 640.0)
                y1 = (yc - h / 2) * (orig_h / 640.0)
                x2 = (xc + w / 2) * (orig_w / 640.0)
                y2 = (yc + h / 2) * (orig_h / 640.0)
                boxes.append([x1, y1, x2, y2])
                scores.append(score)

        return np.array(boxes), np.array(scores)

    def draw_bboxes(self, video_frames, ball_detections, draw_mode="tracer", max_trail=10):
        """
        Draw ball annotations on video frames.
        draw_mode: "tracer" (high-speed broadcast motion comet tail + glowing halo) or "box"
        """
        # Pre-extract center coordinates for all frames to build continuous trajectory
        ball_history = []
        for b_dict in ball_detections:
            b = b_dict.get(1, [])
            if len(b) == 4 and not np.isnan(b[0]):
                ball_history.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
            else:
                ball_history.append(None)

        output_video_frames = []
        for f_idx, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            if draw_mode == "tracer":
                # Collect recent trail points within window
                pts = []
                for i in range(max(0, f_idx - max_trail), f_idx + 1):
                    if i < len(ball_history) and ball_history[i] is not None:
                        pts.append((i, ball_history[i]))

                if pts:
                    # 1. Draw smooth tapering motion comet tail
                    for k in range(len(pts) - 1):
                        idx1, p1 = pts[k]
                        idx2, p2 = pts[k + 1]
                        # Only connect close consecutive frames to avoid jump artifacts
                        if idx2 - idx1 <= 2:
                            progress = (k + 1) / len(pts)
                            thickness = max(2, int(progress * 5))
                            # Vibrant tennis yellow gradient: (B, G, R)
                            b_val = int(40 + 215 * progress)
                            g_val = 255
                            r_val = int(50 + 205 * (1.0 - progress))
                            color = (b_val, g_val, r_val)
                            cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), color, thickness, cv2.LINE_AA)

                    # 2. Draw glowing halo and solid core at current ball position
                    if ball_history[f_idx] is not None:
                        curr_pt = ball_history[f_idx]
                        cx, cy = int(curr_pt[0]), int(curr_pt[1])

                        # Semi-transparent glowing halo
                        overlay = frame.copy()
                        cv2.circle(overlay, (cx, cy), 12, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(overlay, (cx, cy), 7, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

                        # Crisp inner core
                        cv2.circle(frame, (cx, cy), 6, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (cx, cy), 3, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                for track_id, bbox in ball_dict.items():
                    if len(bbox) == 4 and not np.isnan(bbox[0]):
                        x1, y1, x2, y2 = bbox
                        cv2.putText(
                            frame,
                            f"Ball ID: {track_id}",
                            (int(x1), int(max(15, y1 - 10))),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 255, 255),
                            2
                        )
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)

            output_video_frames.append(frame)

        return output_video_frames