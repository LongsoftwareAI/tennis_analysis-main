import os
import cv2
import pickle
from itertools import islice
import numpy as np
import pandas as pd

class BallTracker:
    """
    Tennis Ball Tracker supporting pure TensorFlow (SavedModel, TFLite) 
    as well as YOLO26 models.
    """
    def __init__(self, model_path, model_type="auto", load_model=True, device=None, aux_yolo_path=None, enable_hybrid=True):
        self.model_path = model_path
        self.model_type = model_type
        self.device = device
        self.aux_yolo_path = aux_yolo_path
        self.enable_hybrid = enable_hybrid
        self.aux_yolo = None
        self.backend = None  # 'tracknet_torch', 'tracknet_v4', 'tf_saved_model', 'tflite', or 'ultralytics'
        self.model = None

        if load_model:
            self._load_model(model_path, model_type=model_type)

    def _load_model(self, model_path, model_type="auto"):
        if not model_path:
            raise ValueError("Model path must be specified.")

        # Check if file exists in models/ folder
        if not os.path.exists(str(model_path)):
            candidate = os.path.join('models', os.path.basename(str(model_path)))
            if os.path.exists(candidate):
                model_path = candidate

        # 1. Check if model is PyTorch TrackNet (.pth or .pt with tracknet in name/model_type)
        if (
            str(model_path).endswith('.pth') or 
            (str(model_path).endswith('.pt') and ('tracknet' in str(model_path).lower() or model_type == 'tracknet'))
        ) and os.path.exists(model_path):
            from .tracknet_torch import load_tracknet_model
            import torch
            self.backend = 'tracknet_torch'
            self.target_size = (360, 640)
            self.conf_thresh = 80
            self.device_str = self.device if self.device else ('cuda' if torch.cuda.is_available() else 'cpu')
            print(f"[BallTracker] Loading PyTorch TrackNet model from: {model_path} onto {self.device_str}...")
            self.model = load_tracknet_model(model_path, device=self.device_str)
            print(f"[BallTracker] Loaded PyTorch TrackNet model successfully! (conf_thresh={self.conf_thresh})")

        # 2. Check if model_path is a TrackNetV4 Keras model (.keras)
        elif str(model_path).endswith('.keras') and os.path.exists(model_path):
            import keras
            self.backend = 'tracknet_v4'
            self.target_size = (288, 512)
            self.conf_thresh = 0.30
            print(f"[BallTracker] Loading TrackNetV4 Keras model from: {model_path}...")
            self.model = keras.models.load_model(model_path, compile=False)
            print(f"[BallTracker] Loaded TrackNetV4 model successfully! (conf_thresh={self.conf_thresh})")

        # 2. Check if model_path is a TensorFlow TFLite file
        elif str(model_path).endswith('.tflite') and os.path.exists(model_path):
            import tensorflow as tf
            self.tf = tf
            self.backend = 'tflite'
            self.interpreter = tf.lite.Interpreter(model_path=model_path)
            self.interpreter.allocate_tensors()
            self.input_details = self.interpreter.get_input_details()
            self.output_details = self.interpreter.get_output_details()
            print(f"[BallTracker] Loaded TensorFlow Lite model: {model_path}")

        # 3. Check if model_path is a TensorFlow SavedModel directory
        elif os.path.isdir(model_path) and (
            os.path.exists(os.path.join(model_path, "saved_model.pb")) or 
            os.path.exists(os.path.join(model_path, "fingerprint.pb"))
        ):
            import tensorflow as tf
            self.tf = tf
            self.backend = 'tf_saved_model'
            self.tf_model = tf.saved_model.load(model_path)
            self.infer_fn = self.tf_model.signatures["serving_default"]
            print(f"[BallTracker] Loaded TensorFlow SavedModel from: {model_path}")

        # 4. Otherwise load with YOLO (e.g., YOLO26: ball_detector_yolo26_best.pt)
        else:
            try:
                from ultralytics import YOLO
                self.backend = 'ultralytics'
                self.model = YOLO(model_path)
                print(f"[BallTracker] Loaded YOLO model: {model_path}")
            except Exception as e:
                print(f"[BallTracker] Fallback loading model {model_path}: {e}")

        # 5. Hybrid Fusion Engine: Load auxiliary YOLO model to recover missing TrackNet frames
        if self.backend in ('tracknet_torch', 'tracknet_v4') and self.enable_hybrid:
            yolo_cand = self.aux_yolo_path or "models/ball_detector_yolo26_best.pt"
            if not os.path.exists(str(yolo_cand)):
                cand_in_models = os.path.join('models', os.path.basename(str(yolo_cand)))
                if os.path.exists(cand_in_models):
                    yolo_cand = cand_in_models
            if os.path.exists(str(yolo_cand)):
                try:
                    from ultralytics import YOLO
                    self.aux_yolo = YOLO(yolo_cand)
                    print(f"[BallTracker] Hybrid Engine: Loaded auxiliary YOLO model ({yolo_cand}) for missing frame recovery.")
                except Exception as e:
                    print(f"[BallTracker] Hybrid Engine: Could not load auxiliary YOLO ({e}), continuing with TrackNet only.")
                    self.aux_yolo = None

    def interpolate_ball_positions(self, ball_positions):
        """
        Clean erratic teleportation noise using tracklet velocity clustering,
        followed by physically-constrained linear interpolation.
        """
        ball_positions_list = [x.get(1, []) if (1 in x and len(x[1]) == 4) else [np.nan, np.nan, np.nan, np.nan] for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        n = len(df_ball_positions)

        mid_x = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2.0
        mid_y = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2.0

        # 1. Height filter: reject detections high in stadium roof/audience
        for i in range(n):
            if pd.notna(mid_y.iloc[i]) and mid_y.iloc[i] < 120.0:
                df_ball_positions.iloc[i] = [np.nan, np.nan, np.nan, np.nan]

        mid_x = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2.0
        mid_y = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2.0

        # 2. Tracklet clustering & short isolated noise rejection
        tracklets = []
        curr = []
        for i in range(n):
            if pd.isna(mid_x.iloc[i]):
                continue
            pt = np.array([mid_x.iloc[i], mid_y.iloc[i]])
            if not curr:
                curr.append((i, pt))
            else:
                last_f, last_pt = curr[-1]
                dt = i - last_f
                dist = np.linalg.norm(pt - last_pt)
                # Maximum physical speed of a tennis ball in 25-30fps broadcast is ~55 px/frame
                if dt <= 3 and (dist / dt) <= 55.0:
                    curr.append((i, pt))
                else:
                    tracklets.append(curr)
                    curr = [(i, pt)]
        if curr:
            tracklets.append(curr)

        # Eliminate short isolated noise bursts (<= 3 frames) that are disconnected from rally
        rejected_count = 0
        for tr in tracklets:
            if len(tr) <= 3:
                for f_idx, pt in tr:
                    df_ball_positions.iloc[f_idx] = [np.nan, np.nan, np.nan, np.nan]
                    rejected_count += 1

        if rejected_count > 0:
            print(f"[BallTracker] Trajectory Sanitizer: Suppressed {rejected_count} erratic noise/teleportation detections.")

        # 3. Controlled linear interpolation: ONLY bridge short occlusion gaps (max 3 frames ~ 0.12s)
        df_ball_positions = df_ball_positions.interpolate(method='linear', limit=3)

        # 4. Final velocity gate: eliminate any remaining jump artifact (> 65 px/frame)
        mid_x = (df_ball_positions['x1'] + df_ball_positions['x2']) / 2.0
        mid_y = (df_ball_positions['y1'] + df_ball_positions['y2']) / 2.0
        prev_valid_f = None
        prev_valid_pt = None
        for i in range(n):
            if pd.notna(mid_x.iloc[i]):
                pt = np.array([mid_x.iloc[i], mid_y.iloc[i]])
                if prev_valid_pt is not None:
                    dt = i - prev_valid_f
                    dist = np.linalg.norm(pt - prev_valid_pt)
                    if (dist / dt) > 65.0:
                        df_ball_positions.iloc[i] = [np.nan, np.nan, np.nan, np.nan]
                        continue
                prev_valid_f = i
                prev_valid_pt = pt

        # Normalize bounding box around center (cx, cy)
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

        interpolated_positions = []
        for row in df_ball_positions.to_numpy():
            if pd.notna(row[0]):
                interpolated_positions.append({1: row.tolist()})
            else:
                interpolated_positions.append({})
        return interpolated_positions

    def get_ball_shot_frames(self, ball_positions, player_positions=None):
        """
        Detect frames where a shot occurred based on vertical trajectory inflection.
        Also detects unclosed final return shots near the end of a rally using player proximity.
        """
        if any(not b.get(1) for b in ball_positions):
            ball_positions = self.interpolate_ball_positions(ball_positions)

        ball_positions_list = [x.get(1, []) for x in ball_positions]
        df_ball_positions = pd.DataFrame(ball_positions_list, columns=['x1', 'y1', 'x2', 'y2'])
        df_ball_positions = df_ball_positions.interpolate(limit=4)

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
                        for p_id, p_bbox in p_dict.items():
                            if len(p_bbox) == 4:
                                p_center = ((p_bbox[0] + p_bbox[2]) / 2.0, (p_bbox[1] + p_bbox[3]) / 2.0)
                                dist = np.hypot(bx - p_center[0], by - p_center[1])
                                if dist < min_p_dist:
                                    min_p_dist = dist
                    # If ball is far away from all players, it is an airborne apex or flight turning point, not a shot
                    if min_p_dist > 250.0:
                        continue

                # Active post-shot motion check: a true shot travels rapidly across the court.
                # If subsequent positions remain essentially stationary (span < 18px or mean speed < 3.5 px/f), ball is dead or slow pre-serve bounce.
                subsequent_pts = [
                    (df_ball_positions['mid_x'].iloc[f_chk], df_ball_positions['mid_y'].iloc[f_chk])
                    for f_chk in range(i + 3, min(len(df_ball_positions), i + 12))
                    if pd.notna(df_ball_positions['mid_x'].iloc[f_chk]) and pd.notna(df_ball_positions['mid_y'].iloc[f_chk])
                ]
                if len(subsequent_pts) >= 4:
                    sub_xs = [pt[0] for pt in subsequent_pts]
                    sub_ys = [pt[1] for pt in subsequent_pts]
                    speeds = [np.hypot(subsequent_pts[k+1][0] - subsequent_pts[k][0], subsequent_pts[k+1][1] - subsequent_pts[k][1]) for k in range(len(subsequent_pts) - 1)]
                    mean_spd = np.mean(speeds) if speeds else 0.0
                    if (max(sub_xs) - min(sub_xs) < 18.0 and max(sub_ys) - min(sub_ys) < 18.0) or mean_spd < 3.5:
                        continue

                # Flight direction check: a real hit must travel across the net towards opponent's court
                subsequent_ys = [df_ball_positions['mid_y'].iloc[f_chk] for f_chk in range(i + 4, min(len(df_ball_positions), i + 25)) if pd.notna(df_ball_positions['mid_y'].iloc[f_chk])]
                if subsequent_ys:
                    dy_dir = np.mean(subsequent_ys) - by
                    if by > 500.0 and dy_dir >= -25.0:
                        continue
                    if by <= 500.0 and dy_dir <= 25.0:
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

        # Multi-pass shot recovery:
        # Detect intermediate volleys, overhead smashes, or decisive unclosed winning returns
        # where the ball was struck out of the air or had < 25 frames of sustained delta_y sign,
        # with gap tolerance for high-speed balls.
        added = True
        while added:
            added = False
            candidates = []
            for i in range(20, len(df_ball_positions) - 8):
                if any(abs(i - s) < 22 for s in filtered_shots):
                    continue
                bx, by = df_ball_positions['mid_x'].iloc[i], df_ball_positions['mid_y'].iloc[i]
                if np.isnan(bx) or np.isnan(by) or by < 120.0:
                    continue

                min_dist = float('inf')
                if player_positions is not None:
                    for cf in range(max(0, i - 6), min(len(player_positions), i + 7)):
                        p_dict = player_positions[cf]
                        for pid, pb in p_dict.items():
                            if len(pb) == 4:
                                d = np.hypot(bx - (pb[0] + pb[2]) / 2.0, by - (pb[1] + pb[3]) / 2.0)
                                if d < min_dist:
                                    min_dist = d
                if min_dist > 180.0:
                    continue

                prev_idx = df_ball_positions['mid_x'].iloc[:i].last_valid_index()
                next_idx = df_ball_positions['mid_x'].iloc[i+1:].first_valid_index()
                if next_idx is None or (next_idx - i > 15):
                    continue

                is_hit = False
                impulse = 0.0

                # Branch A: Continuous trajectory inflection
                if prev_idx is not None and (i - prev_idx <= 6):
                    dt_pre = float(i - prev_idx)
                    dt_post = float(next_idx - i)
                    vx_pre = (bx - df_ball_positions['mid_x'].iloc[prev_idx]) / dt_pre
                    vy_pre = (by - df_ball_positions['mid_y'].iloc[prev_idx]) / dt_pre
                    vx_post = (df_ball_positions['mid_x'].iloc[next_idx] - bx) / dt_post
                    vy_post = (df_ball_positions['mid_y'].iloc[next_idx] - by) / dt_post

                    impulse = np.hypot(vx_post - vx_pre, vy_post - vy_pre)
                    ang_pre = np.arctan2(vy_pre, vx_pre)
                    ang_post = np.arctan2(vy_post, vx_post)
                    dang = abs((ang_post - ang_pre + np.pi) % (2 * np.pi) - np.pi)

                    if impulse >= 15.0 and dang >= np.radians(35.0):
                        is_hit = True
                    elif impulse >= 22.0:
                        is_hit = True
                    elif abs(vy_post - vy_pre) >= 12.0 and min_dist < 120.0:
                        is_hit = True

                # Branch B: Emergent return shot after occlusion gap near player
                elif prev_idx is None or (i - prev_idx > 6):
                    subsequent_ys = [df_ball_positions['mid_y'].iloc[f_chk] for f_chk in range(i, min(len(df_ball_positions), i + 14)) if pd.notna(df_ball_positions['mid_y'].iloc[f_chk])]
                    if len(subsequent_ys) >= 4 and min_dist < 150.0:
                        dy_dir = subsequent_ys[-1] - subsequent_ys[0]
                        if (by < 450.0 and dy_dir > 20.0) or (by >= 450.0 and dy_dir < -20.0):
                            is_hit = True
                            impulse = 30.0

                if is_hit:
                    # Active motion check in Pass 2: reject dead ball or stationary points
                    subsequent_pts = [
                        (df_ball_positions['mid_x'].iloc[f_chk], df_ball_positions['mid_y'].iloc[f_chk])
                        for f_chk in range(i + 3, min(len(df_ball_positions), i + 12))
                        if pd.notna(df_ball_positions['mid_x'].iloc[f_chk]) and pd.notna(df_ball_positions['mid_y'].iloc[f_chk])
                    ]
                    if len(subsequent_pts) >= 4:
                        sub_xs = [pt[0] for pt in subsequent_pts]
                        sub_ys = [pt[1] for pt in subsequent_pts]
                        speeds = [np.hypot(subsequent_pts[k+1][0] - subsequent_pts[k][0], subsequent_pts[k+1][1] - subsequent_pts[k][1]) for k in range(len(subsequent_pts) - 1)]
                        mean_spd = np.mean(speeds) if speeds else 0.0
                        if (max(sub_xs) - min(sub_xs) < 18.0 and max(sub_ys) - min(sub_ys) < 18.0) or mean_spd < 3.5:
                            continue

                    subsequent_ys = [df_ball_positions['mid_y'].iloc[f_chk] for f_chk in range(i + 4, min(len(df_ball_positions), i + 25)) if pd.notna(df_ball_positions['mid_y'].iloc[f_chk])]
                    if subsequent_ys:
                        dy_dir = np.mean(subsequent_ys) - by
                        if (by > 500.0 and dy_dir < -25.0) or (by <= 500.0 and dy_dir > 25.0):
                            candidates.append((impulse, i))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                best_f = candidates[0][1]
                filtered_shots.append(best_f)
                filtered_shots.sort()
                added = True

        return filtered_shots

    def _recover_missing_frames_with_yolo(self, frames_list, tracked_boxes, orig_w, orig_h):
        """
        Recover ball detections on frames where TrackNet missed (e.g. during player strikes,
        high-speed smashes, or racket/net occlusions) using kinematic-gated auxiliary YOLO inference.
        """
        if self.aux_yolo is None:
            return tracked_boxes

        total_frames = len(tracked_boxes)
        missing_indices = [i for i, b in enumerate(tracked_boxes) if b is None]
        if not missing_indices:
            return tracked_boxes

        print(f"[BallTracker] Hybrid Fusion: TrackNet missed {len(missing_indices)}/{total_frames} frames. "
              f"Running auxiliary YOLO recovery with kinematic trajectory gating...")

        device_arg = self.device_str if hasattr(self, 'device_str') and self.device_str else (self.device or 'cpu')
        recovered_count = 0

        for idx in missing_indices:
            prev_idx = None
            for p in range(idx - 1, -1, -1):
                if tracked_boxes[p] is not None:
                    prev_idx = p
                    break

            next_idx = None
            for n in range(idx + 1, total_frames):
                if tracked_boxes[n] is not None:
                    next_idx = n
                    break

            # If the gap is too large (> 14 frames ~ 0.5s), don't interpolate blindly with YOLO
            if prev_idx is None and next_idx is None:
                continue

            frame = frames_list[idx]
            res = self.aux_yolo.predict(frame, conf=0.18, verbose=False, device=device_arg)
            boxes = res[0].boxes
            if len(boxes) == 0:
                continue

            cands = []
            for b in boxes:
                coords = b.xyxy.tolist()[0]
                conf = float(b.conf[0])
                cx = (coords[0] + coords[2]) / 2.0
                cy = (coords[1] + coords[3]) / 2.0
                cands.append((cx, cy, conf, coords))

            best_cand = None

            # Case A: Bounded between prev_idx and next_idx
            if prev_idx is not None and next_idx is not None and (next_idx - prev_idx) <= 14:
                p_box = tracked_boxes[prev_idx]
                n_box = tracked_boxes[next_idx]
                px, py = (p_box[0] + p_box[2]) / 2.0, (p_box[1] + p_box[3]) / 2.0
                nx, ny = (n_box[0] + n_box[2]) / 2.0, (n_box[1] + n_box[3]) / 2.0
                alpha = (idx - prev_idx) / float(next_idx - prev_idx)
                ex = px + alpha * (nx - px)
                ey = py + alpha * (ny - py)
                span_dist = np.hypot(nx - px, ny - py)
                gate_radius = max(75.0, 1.4 * span_dist * alpha * (1.0 - alpha) + 40.0)

                valid_cands = [c for c in cands if np.hypot(c[0] - ex, c[1] - ey) <= gate_radius]
                if valid_cands:
                    best_cand = min(valid_cands, key=lambda c: np.hypot(c[0] - ex, c[1] - ey))

            # Case B: Trailing detection (only prev_idx known, within 5 frames)
            elif prev_idx is not None and (idx - prev_idx) <= 5:
                p_box = tracked_boxes[prev_idx]
                px, py = (p_box[0] + p_box[2]) / 2.0, (p_box[1] + p_box[3]) / 2.0
                valid_cands = [c for c in cands if np.hypot(c[0] - px, c[1] - py) <= 85.0 * (idx - prev_idx)]
                if valid_cands:
                    best_cand = max(valid_cands, key=lambda c: c[2])

            # Case C: Leading detection (only next_idx known, within 5 frames)
            elif next_idx is not None and (next_idx - idx) <= 5:
                n_box = tracked_boxes[next_idx]
                nx, ny = (n_box[0] + n_box[2]) / 2.0, (n_box[1] + n_box[3]) / 2.0
                valid_cands = [c for c in cands if np.hypot(c[0] - nx, c[1] - ny) <= 85.0 * (next_idx - idx)]
                if valid_cands:
                    best_cand = max(valid_cands, key=lambda c: c[2])

            if best_cand is not None:
                cx, cy, conf, coords = best_cand
                tracked_boxes[idx] = [cx - 9.0, cy - 9.0, cx + 9.0, cy + 9.0]
                recovered_count += 1

        total_det = sum(1 for b in tracked_boxes if b is not None)
        print(f"[BallTracker] Hybrid Fusion: Successfully recovered {recovered_count}/{len(missing_indices)} missing frames! "
              f"Overall detections: {total_det}/{total_frames} ({total_det / total_frames * 100:.1f}% raw detection rate).")

        return tracked_boxes

    def detect_frames(self, frames, read_from_stub=False, stub_path=None, batch_size=1):
        """
        Run robust ball detection and tracking on a list of video frames,
        with static background noise suppression and directional momentum vector gating.
        """
        if read_from_stub and stub_path is not None and os.path.exists(stub_path):
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        if self.backend == 'tracknet_torch':
            print(f"[BallTracker] Running TrackNet PyTorch triplet inference on {self.device_str}...")
            import torch
            frames_list = list(frames)
            total_frames = len(frames_list)
            if total_frames == 0:
                return []

            orig_h, orig_w = frames_list[0].shape[:2]
            target_h, target_w = self.target_size
            tracked_boxes = [None] * total_frames

            # Pre-resize all frames to 640x360 normalized float32
            resized_frames = []
            for f in frames_list:
                r_img = cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                resized_frames.append(r_img)

            infer_batch_size = max(1, min(4, batch_size if batch_size > 1 else 2))
            num_batches = (total_frames + infer_batch_size - 1) // infer_batch_size

            for b in range(num_batches):
                start_i = b * infer_batch_size
                end_i = min(total_frames, (b + 1) * infer_batch_size)
                cur_b_size = end_i - start_i

                triplets = np.zeros((cur_b_size, 9, target_h, target_w), dtype=np.float32)
                for i_offset, f_idx in enumerate(range(start_i, end_i)):
                    idx_p = max(0, f_idx - 1)
                    idx_c = f_idx
                    idx_n = min(total_frames - 1, f_idx + 1)
                    stacked = np.concatenate([
                        resized_frames[idx_p],
                        resized_frames[idx_c],
                        resized_frames[idx_n]
                    ], axis=2)
                    triplets[i_offset] = np.transpose(stacked, (2, 0, 1))

                batch_tensor = torch.from_numpy(triplets).to(self.device_str)
                with torch.no_grad():
                    out = self.model(batch_tensor)
                    argmax_maps = out.argmax(dim=1).to(torch.uint8).cpu().numpy()
                    del out
                    del batch_tensor

                for i_offset, f_idx in enumerate(range(start_i, end_i)):
                    pred_map = argmax_maps[i_offset]
                    max_conf = int(pred_map.max())
                    if max_conf >= self.conf_thresh:
                        py, px = np.unravel_index(np.argmax(pred_map), pred_map.shape)
                        cx = (px / float(target_w)) * orig_w
                        cy = (py / float(target_h)) * orig_h
                        tracked_boxes[f_idx] = [cx - 9.0, cy - 9.0, cx + 9.0, cy + 9.0]
                    else:
                        tracked_boxes[f_idx] = None

                if self.device_str == 'cuda' and b % 10 == 0:
                    torch.cuda.empty_cache()

            # Hybrid Fusion: Fill missing TrackNet frames via kinematic-gated auxiliary YOLO
            if self.aux_yolo is not None:
                tracked_boxes = self._recover_missing_frames_with_yolo(frames_list, tracked_boxes, orig_w, orig_h)

            detected_count = sum(1 for b in tracked_boxes if b is not None)
            print(f"[BallTracker] TrackNet PyTorch detected ball in {detected_count}/{total_frames} frames ({detected_count/total_frames*100:.1f}% raw detection rate).")

            ball_detections = [{1: b} if b is not None else {} for b in tracked_boxes]
            if stub_path is not None:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)
                print(f"[BallTracker] Saved TrackNet PyTorch detections to stub: {stub_path}")

            return ball_detections

        if self.backend == 'tracknet_v4':
            print("[BallTracker] Running TrackNetV4 temporal triplet inference...")
            from keras import ops
            frames_list = list(frames)
            total_frames = len(frames_list)
            if total_frames == 0:
                return []

            orig_h, orig_w = frames_list[0].shape[:2]
            target_h, target_w = self.target_size
            tracked_boxes = [None] * total_frames

            # Pre-resize all frames to target_size in RGB to optimize batch processing
            resized_frames = []
            for f in frames_list:
                rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                r_img = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
                resized_frames.append(r_img)

            infer_batch_size = max(1, batch_size if batch_size > 1 else 4)
            num_batches = (total_frames + infer_batch_size - 1) // infer_batch_size

            for b in range(num_batches):
                start_i = b * infer_batch_size
                end_i = min(total_frames, (b + 1) * infer_batch_size)
                cur_b_size = end_i - start_i

                X_batch = np.zeros((cur_b_size, target_h, target_w, 9), dtype=np.float32)
                for i_offset, f_idx in enumerate(range(start_i, end_i)):
                    idx_p = max(0, f_idx - 1)
                    idx_c = f_idx
                    idx_n = min(total_frames - 1, f_idx + 1)
                    X_batch[i_offset] = np.concatenate([
                        resized_frames[idx_p],
                        resized_frames[idx_c],
                        resized_frames[idx_n]
                    ], axis=-1)

                preds = self.model.predict_on_batch(X_batch)
                heatmaps = ops.convert_to_numpy(preds['ball_heatmap'])

                for i_offset, f_idx in enumerate(range(start_i, end_i)):
                    pred_map = heatmaps[i_offset, :, :, 0]
                    max_conf = float(pred_map.max())
                    if max_conf >= self.conf_thresh:
                        py, px = np.unravel_index(np.argmax(pred_map), pred_map.shape)
                        cx = (px / float(target_w)) * orig_w
                        cy = (py / float(target_h)) * orig_h
                        tracked_boxes[f_idx] = [cx - 9.0, cy - 9.0, cx + 9.0, cy + 9.0]
                    else:
                        tracked_boxes[f_idx] = None

            # Hybrid Fusion: Fill missing TrackNetV4 frames via kinematic-gated auxiliary YOLO
            if self.aux_yolo is not None:
                tracked_boxes = self._recover_missing_frames_with_yolo(frames_list, tracked_boxes, orig_w, orig_h)

            detected_count = sum(1 for b in tracked_boxes if b is not None)
            print(f"[BallTracker] TrackNetV4 detected ball in {detected_count}/{total_frames} frames ({detected_count/total_frames*100:.1f}% raw detection rate).")

            ball_detections = [{1: b} if b is not None else {} for b in tracked_boxes]
            if stub_path is not None:
                os.makedirs(os.path.dirname(stub_path), exist_ok=True)
                with open(stub_path, 'wb') as f:
                    pickle.dump(ball_detections, f)
                print(f"[BallTracker] Saved TrackNetV4 detections to stub: {stub_path}")

            return ball_detections

        print("[BallTracker] Running candidate extraction...")
        # 1. Extract raw candidate detections for each frame
        raw_candidates_per_frame = []
        if self.backend == 'ultralytics' and batch_size > 1:
            frame_iter = iter(frames)
            predict_kwargs = {'conf': 0.10, 'verbose': False}
            if self.device is not None:
                predict_kwargs['device'] = self.device
            while batch := list(islice(frame_iter, batch_size)):
                for result in self.model.predict(batch, **predict_kwargs):
                    cands = []
                    for box in result.boxes:
                        coords = box.xyxy.tolist()[0]
                        score = float(box.conf[0])
                        cx = (coords[0] + coords[2]) / 2.0
                        cy = (coords[1] + coords[3]) / 2.0
                        cands.append({'box': coords, 'conf': score, 'center': np.array([cx, cy])})
                    raw_candidates_per_frame.append(cands)
        else:
            for frame in frames:
                raw_candidates_per_frame.append(self._detect_candidates(frame, conf=0.10))
        total_frames = len(raw_candidates_per_frame)

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
        tracked_boxes = [None] * total_frames
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
        for f_idx in range(total_frames - 2, -1, -1):
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
            predict_kwargs = {'conf': conf, 'verbose': False}
            if self.device is not None:
                predict_kwargs['device'] = self.device
            results = self.model.predict(frame, **predict_kwargs)[0]
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
                input_tensor = self.tf.convert_to_tensor(img_rgb[np.newaxis, ...], dtype=self.tf.float32) / 255.0
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
        elif self.backend == 'tracknet_torch':
            import torch
            h, w = frame.shape[:2]
            target_h, target_w = self.target_size
            r_img = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            stacked = np.concatenate([r_img, r_img, r_img], axis=2)
            inp = np.transpose(stacked, (2, 0, 1))
            t = torch.from_numpy(inp).unsqueeze(0).to(self.device_str)
            with torch.no_grad():
                out = self.model(t)
                am = out.argmax(dim=1).cpu().numpy()[0]
                max_conf = int(am.max())
                thresh = self.conf_thresh if conf <= 0.15 else int(conf * 255)
                if max_conf >= thresh:
                    py, px = np.unravel_index(np.argmax(am), am.shape)
                    cx = (px / float(target_w)) * w
                    cy = (py / float(target_h)) * h
                    candidates.append({
                        'box': [cx - 9.0, cy - 9.0, cx + 9.0, cy + 9.0],
                        'conf': float(max_conf) / 255.0,
                        'center': np.array([cx, cy])
                    })
        elif self.backend == 'tracknet_v4':
            from keras import ops
            h, w = frame.shape[:2]
            target_h, target_w = self.target_size
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r_img = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
            X = np.expand_dims(np.concatenate([r_img, r_img, r_img], axis=-1), axis=0)
            preds = self.model(X, training=False)
            pred_map = ops.convert_to_numpy(preds['ball_heatmap'])[0, :, :, 0]
            max_conf = float(pred_map.max())
            if max_conf >= conf:
                py, px = np.unravel_index(np.argmax(pred_map), pred_map.shape)
                cx = (px / float(target_w)) * w
                cy = (py / float(target_h)) * h
                candidates.append({
                    'box': [cx - 9.0, cy - 9.0, cx + 9.0, cy + 9.0],
                    'conf': max_conf,
                    'center': np.array([cx, cy])
                })
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

    def draw_bboxes(self, video_frames, ball_detections, draw_mode="tracer", max_trail=7,
                    start_frame=0, ball_history=None):
        """
        Draw ball annotations on video frames.
        draw_mode: "tracer" (high-speed broadcast motion comet tail + glowing halo) or "box"
        """
        # Pre-extract center coordinates for all frames to build continuous trajectory
        if ball_history is None:
            ball_history = []
            for b_dict in ball_detections:
                b = b_dict.get(1, [])
                if len(b) == 4 and not np.isnan(b[0]):
                    ball_history.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))
                else:
                    ball_history.append(None)

        output_video_frames = []
        for local_idx, (frame, ball_dict) in enumerate(zip(video_frames, ball_detections)):
            f_idx = start_frame + local_idx
            if draw_mode == "tracer":
                # Collect recent trail points within window
                pts = []
                for i in range(max(0, f_idx - max_trail), f_idx + 1):
                    if i < len(ball_history) and ball_history[i] is not None:
                        pts.append((i, ball_history[i]))

                if pts and (pts[-1][0] >= f_idx - 1):
                    # 1. Draw smooth tapering motion comet tail with physical speed gating
                    for k in range(len(pts) - 1):
                        idx1, p1 = pts[k]
                        idx2, p2 = pts[k + 1]
                        dt = idx2 - idx1
                        dist = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                        # Only connect close consecutive frames with physical tennis speed (dist <= 65px * dt and dist >= 1.5px)
                        # Never draw a jump or streak if the ball teleports or was dropped
                        if dt <= 2 and dist <= 65.0 * dt and dist >= 1.5:
                            progress = (k + 1) / len(pts)
                            thickness = max(2, int(progress * 5))
                            # Vibrant tennis yellow gradient: (B, G, R)
                            b_val = int(35 + 200 * progress)
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
                        cv2.circle(overlay, (cx, cy), 11, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(overlay, (cx, cy), 6, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.addWeighted(overlay, 0.40, frame, 0.60, 0, frame)

                        # Crisp inner core
                        cv2.circle(frame, (cx, cy), 5, (0, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1, cv2.LINE_AA)
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
