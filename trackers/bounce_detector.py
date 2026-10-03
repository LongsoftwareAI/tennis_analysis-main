import os
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional

try:
    import catboost as ctb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False


class BounceDetector:
    """
    Machine Learning-based Tennis Ball Bounce Detector using CatBoost.
    Learns kinematic reversal features (V-shape trajectory and velocity ratios)
    over a sliding temporal window to accurately identify ground contact frames.
    """
    def __init__(self, model_path: str = "models/bounce_model.cbm", threshold: float = 0.40):
        self.model_path = model_path
        self.threshold = threshold
        self.model = None

        if HAS_CATBOOST and os.path.exists(model_path):
            try:
                self.model = ctb.CatBoostRegressor()
                self.model.load_model(model_path)
            except Exception as e:
                print(f"[BounceDetector] Warning: Could not load CatBoost model from {model_path}: {e}")
                self.model = None
        else:
            if not HAS_CATBOOST:
                print("[BounceDetector] Warning: 'catboost' library is not installed.")
            elif not os.path.exists(model_path):
                print(f"[BounceDetector] Warning: Model file {model_path} does not exist.")

    def is_available(self) -> bool:
        """Check if CatBoost model is loaded and ready for inference."""
        return self.model is not None

    def _prepare_features(self, x_coords: pd.Series, y_coords: pd.Series) -> Tuple[pd.DataFrame, List[int]]:
        """
        Extract the 12 kinematic features required by CatBoost bounce detection model:
        x/y differential displacements, inverse displacements, and velocity change ratios
        over a sliding window of temporal radius num=3 (frames t-2, t-1, t, t+1, t+2).
        """
        labels = pd.DataFrame({
            'frame': range(len(x_coords)),
            'x-coordinate': x_coords,
            'y-coordinate': y_coords
        })

        num = 3
        eps = 1e-15
        for i in range(1, num):
            labels[f'x_lag_{i}'] = labels['x-coordinate'].shift(i)
            labels[f'x_lag_inv_{i}'] = labels['x-coordinate'].shift(-i)
            labels[f'y_lag_{i}'] = labels['y-coordinate'].shift(i)
            labels[f'y_lag_inv_{i}'] = labels['y-coordinate'].shift(-i)

            labels[f'x_diff_{i}'] = (labels[f'x_lag_{i}'] - labels['x-coordinate']).abs()
            labels[f'y_diff_{i}'] = labels[f'y_lag_{i}'] - labels['y-coordinate']
            labels[f'x_diff_inv_{i}'] = (labels[f'x_lag_inv_{i}'] - labels['x-coordinate']).abs()
            labels[f'y_diff_inv_{i}'] = labels[f'y_lag_inv_{i}'] - labels['y-coordinate']

            labels[f'x_div_{i}'] = (labels[f'x_diff_{i}'] / (labels[f'x_diff_inv_{i}'] + eps)).abs()
            labels[f'y_div_{i}'] = labels[f'y_diff_{i}'] / (labels[f'y_diff_inv_{i}'] + eps)

        for i in range(1, num):
            labels = labels[labels[f'x_lag_{i}'].notna()]
            labels = labels[labels[f'x_lag_inv_{i}'].notna()]
        labels = labels[labels['x-coordinate'].notna()]

        colnames_x = (
            [f'x_diff_{i}' for i in range(1, num)] +
            [f'x_diff_inv_{i}' for i in range(1, num)] +
            [f'x_div_{i}' for i in range(1, num)]
        )
        colnames_y = (
            [f'y_diff_{i}' for i in range(1, num)] +
            [f'y_diff_inv_{i}' for i in range(1, num)] +
            [f'y_div_{i}' for i in range(1, num)]
        )
        colnames = colnames_x + colnames_y

        if len(labels) == 0:
            return pd.DataFrame(columns=colnames), []

        return labels[colnames], list(labels['frame'])

    def detect_bounces(
        self,
        ball_positions: List[Dict],
        min_gap: int = 15,
        confidence_threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect ball bounces from ball tracking positions.

        Args:
            ball_positions: list of frame dicts {1: [x1, y1, x2, y2]} or tuples.
            min_gap: minimum frame distance between two consecutive bounces.
            confidence_threshold: custom threshold override.

        Returns:
            list of bounce dicts:
            [{'frame': int, 'score': float, 'camera_pos': (float, float)}, ...]
        """
        if not self.is_available() or not ball_positions:
            return []

        thresh = confidence_threshold if confidence_threshold is not None else self.threshold

        xs = []
        ys = []
        for pos in ball_positions:
            b = pos.get(1, []) if isinstance(pos, dict) else pos
            if len(b) == 4 and not np.isnan(b[0]):
                xs.append((b[0] + b[2]) / 2.0)
                ys.append((b[1] + b[3]) / 2.0)
            else:
                xs.append(np.nan)
                ys.append(np.nan)

        # Bridge brief occlusions (up to 5 frames) to enable feature computation
        s_x = pd.Series(xs).interpolate(method='linear', limit=5)
        s_y = pd.Series(ys).interpolate(method='linear', limit=5)

        features, valid_frames = self._prepare_features(s_x, s_y)
        if len(features) == 0:
            return []

        try:
            preds = self.model.predict(features)
        except Exception as e:
            print(f"[BounceDetector] Inference error: {e}")
            return []

        candidates = []
        for f, score in zip(valid_frames, preds):
            if score >= thresh:
                candidates.append((f, float(score), float(s_x[f]), float(s_y[f])))

        if not candidates:
            return []

        # Cluster consecutive frames within 4 frames
        clusters = []
        current_cluster = [candidates[0]]
        for cand in candidates[1:]:
            if cand[0] - current_cluster[-1][0] <= 4:
                current_cluster.append(cand)
            else:
                clusters.append(current_cluster)
                current_cluster = [cand]
        clusters.append(current_cluster)

        # Pick argmax score in each cluster
        bounces = []
        for clust in clusters:
            best = max(clust, key=lambda c: c[1])
            bounces.append({
                'frame': int(best[0]),
                'score': float(best[1]),
                'camera_pos': (float(best[2]), float(best[3]))
            })

        # Enforce physical minimum inter-bounce gap
        filtered_bounces = []
        for b in bounces:
            if not filtered_bounces or (b['frame'] - filtered_bounces[-1]['frame'] >= min_gap):
                filtered_bounces.append(b)
            else:
                if b['score'] > filtered_bounces[-1]['score']:
                    filtered_bounces[-1] = b

        return filtered_bounces
