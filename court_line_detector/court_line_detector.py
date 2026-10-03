import os

import cv2
import numpy as np
import torch

from .court_heatmap_model import CourtHeatmapNet
from .court_postprocess import extract_keypoint, reconstruct_keypoints, refine_keypoint


class CourtLineDetector:
    """Detect 14 tennis-court keypoints with TrackNet-style heatmaps."""

    input_width = 640
    input_height = 360
    num_keypoints = 14

    def __init__(self, model_path=None, load_model=True, device=None):
        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

        if load_model:
            if not model_path or not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"Court heatmap weights were not found. Expected '{model_path}'.\n"
                    "👉 Vui lòng tải file trọng số model_tennis_court_det.pt và đặt vào thư mục models/ "
                    "theo hướng dẫn trong README.md."
                )
            self.model = self.build_model()
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)
            self.model.eval()

    def build_model(self):
        return CourtHeatmapNet(out_channels=self.num_keypoints + 1).to(self.device)

    def preprocess_image(self, image):
        resized = cv2.resize(image, (self.input_width, self.input_height))
        normalized = resized.astype(np.float32) / 255.0
        channels_first = np.ascontiguousarray(np.transpose(normalized, (2, 0, 1)))
        return torch.from_numpy(channels_first).unsqueeze(0).to(self.device)

    def predict(self, image):
        if self.model is None:
            raise RuntimeError("CourtLineDetector was initialized without model weights.")

        with torch.inference_mode():
            logits = self.model(self.preprocess_image(image))[0]
            heatmaps = torch.sigmoid(logits).cpu().numpy()

        height, width = image.shape[:2]
        scale_x = width / float(self.input_width)
        scale_y = height / float(self.input_height)
        points = []

        for index in range(self.num_keypoints):
            heatmap = (heatmaps[index] * 255.0).astype(np.uint8)
            point = extract_keypoint(heatmap, scale_x=scale_x, scale_y=scale_y)
            if point is not None and index not in (8, 9, 12):
                point = refine_keypoint(image, point)
            points.append(point)

        reconstructed = reconstruct_keypoints(points)
        if reconstructed is not None:
            points = reconstructed
        elif any(point is None for point in points):
            missing = [index for index, point in enumerate(points) if point is None]
            raise RuntimeError(f"Court detector could not locate keypoints: {missing}")
        else:
            points = np.asarray(points, dtype=np.float32)

        return np.asarray(points, dtype=np.float32).reshape(-1)

    def draw_keypoints(self, image, keypoints):
        output_image = image.copy()
        for index in range(0, len(keypoints), 2):
            x_coord = int(round(keypoints[index]))
            y_coord = int(round(keypoints[index + 1]))
            cv2.putText(
                output_image,
                str(index // 2),
                (x_coord, max(0, y_coord - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
            cv2.circle(output_image, (x_coord, y_coord), 5, (0, 0, 255), -1)
        return output_image

    def draw_keypoints_on_video(self, video_frames, keypoints):
        output_video_frames = []
        is_dynamic = (
            isinstance(keypoints, (list, np.ndarray))
            and len(keypoints) == len(video_frames)
            and len(keypoints) > 0
            and hasattr(keypoints[0], "__len__")
            and len(keypoints[0]) >= self.num_keypoints * 2
        )

        for index, frame in enumerate(video_frames):
            frame_keypoints = keypoints[index] if is_dynamic else keypoints
            output_video_frames.append(self.draw_keypoints(frame, frame_keypoints))
        return output_video_frames

    def save(self, filepath):
        if self.model is None:
            raise RuntimeError("There is no loaded court model to save.")
        torch.save(self.model.state_dict(), filepath)
