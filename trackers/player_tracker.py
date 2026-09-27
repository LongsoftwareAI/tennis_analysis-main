import os
import cv2
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from utils import measure_distance, get_center_of_bbox

class PlayerTracker:
    """
    Tennis Player Tracker supporting TensorFlow models and YOLO26 tracking.
    """
    def __init__(self, model_path='yolo26s.pt'):
        self.model_path = model_path
        self.backend = None
        self.model = None

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

    def choose_and_filter_players(self, court_keypoints, player_detections):
        """
        Robustly choose the two actual tennis players across video frames:
        one on the near side of the net (Player 1) and one on the far side of the net (Player 2).
        Filter out all spectators, line judges, and ball boys.
        """
        if not player_detections or len(player_detections) == 0:
            return player_detections

        # Determine net line and court bounding box dynamically from court keypoints
        ref_kps = court_keypoints[0] if (court_keypoints is not None and len(court_keypoints) > 0 and hasattr(court_keypoints[0], '__len__')) else court_keypoints
        if ref_kps is not None and len(ref_kps) >= 28:
            xs = [ref_kps[i] for i in range(0, len(ref_kps), 2)]
            ys = [ref_kps[i + 1] for i in range(0, len(ref_kps), 2)]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            court_w = max_x - min_x
            court_h = max_y - min_y
            
            # Net line Y is the midpoint between top and bottom service lines
            net_y = (ref_kps[2 * 8 + 1] + ref_kps[2 * 10 + 1]) / 2.0
            
            # Dynamic perimeter with buffer for players behind baseline & outside sidelines
            valid_x_min = min_x - court_w * 0.35
            valid_x_max = max_x + court_w * 0.35
            valid_y_min = min_y - court_h * 0.40
            valid_y_max = max_y + court_h * 0.40
        else:
            net_y = 530.0
            valid_x_min, valid_x_max = 50.0, 1900.0
            valid_y_min, valid_y_max = 50.0, 1050.0

        # Collect track statistics across all frames
        track_stats = {}
        for p_dict in player_detections:
            for track_id, bbox in p_dict.items():
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0

                # Filter out spectators far outside court perimeter
                if cx < valid_x_min or cx > valid_x_max or cy < valid_y_min or cy > valid_y_max:
                    continue

                if track_id not in track_stats:
                    track_stats[track_id] = {'ys': [], 'xs': []}
                track_stats[track_id]['ys'].append(cy)
                track_stats[track_id]['xs'].append(cx)

        # Classify candidate tracks for Near player (Player 1) and Far player (Player 2)
        # Active players move on the court (dy >= 25), whereas line judges/ball kids are stationary (dy < 20)
        near_candidates = {}
        far_candidates = {}

        for track_id, stats in track_stats.items():
            count = len(stats['ys'])
            if count < 5:
                continue
            avg_y = float(np.mean(stats['ys']))
            dy = max(stats['ys']) - min(stats['ys'])
            
            # An active player has vertical movement (dy >= 25)
            if dy >= 25:
                if avg_y > net_y:
                    near_candidates[track_id] = count
                else:
                    far_candidates[track_id] = count

        # Fallback if no candidate found: pick track with most frames on that side
        if not near_candidates:
            near_by_count = {t: len(s['ys']) for t, s in track_stats.items() if np.mean(s['ys']) > net_y}
            if near_by_count:
                top_near = max(near_by_count.keys(), key=lambda t: near_by_count[t])
                near_candidates[top_near] = near_by_count[top_near]

        if not far_candidates:
            far_by_count = {t: len(s['ys']) for t, s in track_stats.items() if np.mean(s['ys']) <= net_y}
            if far_by_count:
                top_far = max(far_by_count.keys(), key=lambda t: far_by_count[t])
                far_candidates[top_far] = far_by_count[top_far]

        # Primary track on each side (has highest occurrence)
        primary_p1 = max(near_candidates.keys(), key=lambda t: near_candidates[t])
        primary_p2 = max(far_candidates.keys(), key=lambda t: far_candidates[t])

        print(f"[PlayerTracker] Near Player (P1): Primary Track {primary_p1}, Candidates: {sorted(list(near_candidates.keys()))}")
        print(f"[PlayerTracker] Far Player (P2): Primary Track {primary_p2}, Candidates: {sorted(list(far_candidates.keys()))}")

        # Frame-by-frame player assignment with temporal continuity & track stitching
        filtered_player_detections = []
        for player_dict in player_detections:
            frame_res = {}
            # Near player (P1): prefer primary_p1, otherwise pick best candidate present
            if primary_p1 in player_dict:
                frame_res[1] = player_dict[primary_p1]
            else:
                p1_cands = [t for t in near_candidates if t in player_dict]
                if p1_cands:
                    best_p1 = max(p1_cands, key=lambda t: near_candidates[t])
                    frame_res[1] = player_dict[best_p1]

            # Far player (P2): prefer primary_p2, otherwise pick best candidate present
            if primary_p2 in player_dict:
                frame_res[2] = player_dict[primary_p2]
            else:
                p2_cands = [t for t in far_candidates if t in player_dict]
                if p2_cands:
                    best_p2 = max(p2_cands, key=lambda t: far_candidates[t])
                    frame_res[2] = player_dict[best_p2]

            filtered_player_detections.append(frame_res)

        return filtered_player_detections

    def interpolate_player_positions(self, player_detections):
        """
        Interpolate missing player detections using linear interpolation and backfill/forward-fill,
        ensuring both players (Player 1 and Player 2) are present in every frame.
        """
        if not player_detections:
            return player_detections

        p1_list = [f.get(1, []) for f in player_detections]
        p2_list = [f.get(2, []) for f in player_detections]

        df_p1 = pd.DataFrame([x if len(x) == 4 else [np.nan] * 4 for x in p1_list], columns=['x1', 'y1', 'x2', 'y2'])
        df_p2 = pd.DataFrame([x if len(x) == 4 else [np.nan] * 4 for x in p2_list], columns=['x1', 'y1', 'x2', 'y2'])

        df_p1 = df_p1.interpolate().bfill().ffill()
        df_p2 = df_p2.interpolate().bfill().ffill()

        p1_boxes = df_p1.to_numpy().tolist()
        p2_boxes = df_p2.to_numpy().tolist()

        interpolated_detections = []
        for i in range(len(player_detections)):
            d = {}
            if not np.isnan(p1_boxes[i][0]):
                d[1] = p1_boxes[i]
            if not np.isnan(p2_boxes[i][0]):
                d[2] = p2_boxes[i]
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
            results = self.model.track(frame, persist=True, verbose=False)[0]
            id_name_dict = results.names

            for box in results.boxes:
                if box.id is not None:
                    track_id = int(box.id.tolist()[0])
                else:
                    track_id = 0
                result = box.xyxy.tolist()[0]
                object_cls_id = int(box.cls.tolist()[0])
                object_cls_name = id_name_dict.get(object_cls_id, "")
                if object_cls_name == "person":
                    player_dict[track_id] = result

        elif self.backend == 'tf_saved_model':
            # Preprocess frame for TensorFlow model
            h, w = frame.shape[:2]
            img_resized = cv2.resize(frame, (640, 640))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
            input_tensor = tf.convert_to_tensor(img_rgb[np.newaxis, ...], dtype=tf.float32) / 255.0

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

    def draw_bboxes(self, video_frames, player_detections):
        """
        Draw player bounding boxes and IDs on video frames.
        """
        output_video_frames = []
        for frame, player_dict in zip(video_frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(
                    frame,
                    f"Player ID: {track_id}",
                    (int(x1), int(max(15, y1 - 10))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 0, 255),
                    2
                )
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            output_video_frames.append(frame)

        return output_video_frames