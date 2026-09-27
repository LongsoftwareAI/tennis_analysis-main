import os
# Đảm bảo Keras 3 sử dụng backend PyTorch để tăng tốc 100% bằng GPU RTX 5060 (CUDA)
if "KERAS_BACKEND" not in os.environ:
    os.environ["KERAS_BACKEND"] = "torch"

import cv2
import numpy as np
import keras
from keras import layers

class CourtLineDetector:
    """
    Tennis Court Line and Keypoints Detector implemented in Keras 3 (PyTorch / GPU accelerated).
    Uses a ResNet50 backbone to regress 14 2D court keypoints (28 coordinate values).
    """
    def __init__(self, model_path=None):
        self.num_keypoints = 14
        self.input_shape = (224, 224, 3)

        if model_path and os.path.exists(model_path):
            try:
                # Attempt to load full Keras model
                self.model = keras.models.load_model(model_path, compile=False)
            except Exception:
                # Fallback to building architecture and loading weights
                self.model = self.build_model()
                self.model.load_weights(model_path)
        else:
            self.model = self.build_model()

    def build_model(self):
        """
        Build ResNet50 regression model for court keypoints detection.
        """
        base_model = keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=self.input_shape
        )
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.2)(x)
        outputs = layers.Dense(self.num_keypoints * 2, activation="linear", name="keypoints_output")(x)
        
        model = keras.Model(inputs=base_model.input, outputs=outputs, name="CourtLineDetector_TF")
        return model

    def preprocess_image(self, image):
        """
        Convert BGR image to RGB, resize to 224x224 and normalize using ImageNet statistics.
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, (self.input_shape[1], self.input_shape[0]))
        
        # Normalize to [0, 1] then standardize with ImageNet mean and std
        norm_img = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        norm_img = (norm_img - mean) / std

        # Add batch dimension: (1, 224, 224, 3)
        return np.expand_dims(norm_img, axis=0)

    def predict(self, image):
        """
        Predict 14 court keypoints for a given frame.
        Returns coordinates scaled to original image dimensions: [x0, y0, x1, y1, ...]
        """
        input_tensor = self.preprocess_image(image)
        # Fast inference using tensor call without graph tracing overhead
        preds = self.model(input_tensor, training=False)
        keypoints = keras.ops.convert_to_numpy(preds).squeeze()

        original_h, original_w = image.shape[:2]
        keypoints = keypoints.copy()
        # Scale back from 224x224 to original frame size
        keypoints[::2] *= original_w / float(self.input_shape[1])
        keypoints[1::2] *= original_h / float(self.input_shape[0])

        # Automatically refine keypoints by snapping to actual court white lines
        keypoints = self.refine_court_keypoints(image, keypoints)

        return keypoints

    def refine_court_keypoints(self, image, coarse_kps):
        """
        Coarse-to-Fine Sub-pixel Refinement:
        Takes initial coarse keypoint predictions from ResNet50 and dynamically snaps them
        onto physical white court line intersections using local white-line profiling.
        """
        try:
            h, w = image.shape[:2]
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower_white = np.array([0, 0, 180])
            upper_white = np.array([180, 50, 255])
            mask = cv2.inRange(hsv, lower_white, upper_white)

            # Restrict to tennis court area to remove spectator / background / billboard noise
            court_mask = np.zeros_like(mask)
            court_mask[int(h * 0.23): int(h * 0.95), int(w * 0.05): int(w * 0.95)] = 255
            lines_mask = cv2.bitwise_and(mask, court_mask)

            proj_y = np.sum(lines_mask > 0, axis=1)

            # 1. Refine the 4 horizontal court levels guided by ResNet50 coarse anchors
            h_groups = {
                'far_base': [0, 1, 4, 6],
                'far_serv': [8, 9, 12],
                'near_serv': [10, 11, 13],
                'near_base': [2, 3, 5, 7]
            }

            refined_y = {}
            for name, indices in h_groups.items():
                anchor_y = np.mean([coarse_kps[2 * i + 1] for i in indices])
                win = 65  # Search +-65 pixels around ResNet anchor to adapt across tournaments
                floor_y = int(h * 0.23) if name == 'far_base' else 0
                y_min = max(floor_y, int(anchor_y - win))
                y_max = min(h, int(anchor_y + win))
                sub = proj_y[y_min:y_max]
                if len(sub) > 0 and np.max(sub) > 40:
                    peak = y_min + np.argmax(sub)
                    refined_y[name] = float(peak)
                else:
                    refined_y[name] = float(anchor_y)

            # 2. Refine the 5 longitudinal lines (vertical/diagonal) guided by ResNet50 line paths
            v_pairs = [
                (0, 2),   # left doubles
                (4, 5),   # left singles
                (12, 13), # center line
                (6, 7),   # right singles
                (1, 3)    # right doubles
            ]

            refined_v = []
            for top_idx, bot_idx in v_pairs:
                top_x, top_y = coarse_kps[2 * top_idx], coarse_kps[2 * top_idx + 1]
                bot_x, bot_y = coarse_kps[2 * bot_idx], coarse_kps[2 * bot_idx + 1]

                y_samples = np.linspace(refined_y['far_base'], refined_y['near_base'], num=16)
                detected_pts = []
                for sy in y_samples:
                    sy_int = int(sy)
                    if sy_int < 0 or sy_int >= h:
                        continue
                    m_coarse = (bot_x - top_x) / (bot_y - top_y) if bot_y != top_y else 0
                    c_coarse = top_x - m_coarse * top_y
                    exp_x = int(m_coarse * sy + c_coarse)

                    win_x = 35
                    xmin, xmax = max(0, exp_x - win_x), min(w, exp_x + win_x)
                    row = lines_mask[sy_int, xmin:xmax]
                    white_xs = np.where(row > 0)[0]
                    if len(white_xs) > 0:
                        detected_pts.append((xmin + np.mean(white_xs), sy))

                if len(detected_pts) >= 4:
                    ys = np.array([p[1] for p in detected_pts])
                    xs = np.array([p[0] for p in detected_pts])
                    poly = np.polyfit(ys, xs, deg=1)
                    refined_v.append((poly[0], poly[1]))
                else:
                    m_coarse = (bot_x - top_x) / (bot_y - top_y) if bot_y != top_y else 0
                    c_coarse = top_x - m_coarse * top_y
                    refined_v.append((m_coarse, c_coarse))

            # 3. Intersect refined horizontal levels with refined longitudinal lines
            ld, ls, c_line, rs, rd = refined_v
            y_fb = refined_y['far_base']
            y_fs = refined_y['far_serv']
            y_ns = refined_y['near_serv']
            y_nb = refined_y['near_base']

            out_kps = np.zeros(28, dtype=np.float32)
            # Point 0: Left doubles at Far Baseline
            out_kps[0] = ld[0] * y_fb + ld[1]; out_kps[1] = y_fb
            # Point 1: Right doubles at Far Baseline
            out_kps[2] = rd[0] * y_fb + rd[1]; out_kps[3] = y_fb
            # Point 2: Left doubles at Near Baseline
            out_kps[4] = ld[0] * y_nb + ld[1]; out_kps[5] = y_nb
            # Point 3: Right doubles at Near Baseline
            out_kps[6] = rd[0] * y_nb + rd[1]; out_kps[7] = y_nb

            # Point 4: Left singles at Far Baseline
            out_kps[8] = ls[0] * y_fb + ls[1]; out_kps[9] = y_fb
            # Point 5: Left singles at Near Baseline
            out_kps[10] = ls[0] * y_nb + ls[1]; out_kps[11] = y_nb
            # Point 6: Right singles at Far Baseline
            out_kps[12] = rs[0] * y_fb + rs[1]; out_kps[13] = y_fb
            # Point 7: Right singles at Near Baseline
            out_kps[14] = rs[0] * y_nb + rs[1]; out_kps[15] = y_nb

            # Point 8: Left singles at Far Service line
            out_kps[16] = ls[0] * y_fs + ls[1]; out_kps[17] = y_fs
            # Point 9: Right singles at Far Service line
            out_kps[18] = rs[0] * y_fs + rs[1]; out_kps[19] = y_fs
            # Point 10: Left singles at Near Service line
            out_kps[20] = ls[0] * y_ns + ls[1]; out_kps[21] = y_ns
            # Point 11: Right singles at Near Service line
            out_kps[22] = rs[0] * y_ns + rs[1]; out_kps[23] = y_ns

            # Point 12: Center service line at Far Service line
            out_kps[24] = c_line[0] * y_fs + c_line[1]; out_kps[25] = y_fs
            # Point 13: Center service line at Near Service line
            out_kps[26] = c_line[0] * y_ns + c_line[1]; out_kps[27] = y_ns

            return out_kps
        except Exception as e:
            print(f"[CourtLineDetector] Line refinement fallback: {e}")
            return coarse_kps

    def draw_keypoints(self, image, keypoints):
        """
        Plot keypoint circles and index numbers on an image.
        """
        output_image = image.copy()
        for i in range(0, len(keypoints), 2):
            x = int(round(keypoints[i]))
            y = int(round(keypoints[i + 1]))
            cv2.putText(
                output_image,
                str(i // 2),
                (x, max(0, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2
            )
            cv2.circle(output_image, (x, y), 5, (0, 0, 255), -1)
        return output_image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        """
        Draw keypoints across all video frames.
        Supports:
        - Static keypoints (shape (28,) or list): drawn uniformly on all frames.
        - Dynamic keypoints (shape (N, 28) or list of frames): drawn per corresponding frame.
        """
        output_video_frames = []
        is_dynamic = False
        if isinstance(keypoints, (list, np.ndarray)) and len(keypoints) == len(video_frames):
            if hasattr(keypoints[0], '__len__') and len(keypoints[0]) >= 28:
                is_dynamic = True

        for i, frame in enumerate(video_frames):
            frame_kps = keypoints[i] if is_dynamic else keypoints
            drawn_frame = self.draw_keypoints(frame, frame_kps)
            output_video_frames.append(drawn_frame)
        return output_video_frames

    def save(self, filepath):
        """
        Save the TensorFlow / Keras model to disk (.keras or .h5).
        """
        self.model.save(filepath)