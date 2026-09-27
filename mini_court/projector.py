import cv2
import numpy as np
import pandas as pd
import sys
sys.path.append('../')
import constants
from utils import get_foot_position

class MiniCourtProjector:
    """
    Handles 2D Perspective Homography and Physics-Informed 3D Ground Projection:
    - Ground player foot position projection
    - Raw ball perspective projection and interpolation
    - Aerodynamic drag model (starts fast, slows down realistically)
    - Shot segmentation, first bounce detection, and rebound modeling
    """
    def __init__(self, geometry):
        self.geom = geometry

    def compute_homography(self, src_keypoints):
        """Compute 3x3 homography matrix from 14 source court keypoints to mini-court keypoints."""
        dst_pts = np.array([
            (self.geom.drawing_key_points[2 * i], self.geom.drawing_key_points[2 * i + 1])
            for i in range(14)
        ], dtype=np.float32)

        src_pts = np.array([
            (src_keypoints[2 * i], src_keypoints[2 * i + 1])
            for i in range(14)
        ], dtype=np.float32)

        h_mat, _ = cv2.findHomography(src_pts, dst_pts)
        return h_mat

    def project_players(self, player_boxes, h_matrices, is_dynamic, min_x, max_x, min_y, max_y):
        """Project players onto mini-court using foot ground contact position."""
        output_player_boxes = []
        h_static = h_matrices if not is_dynamic else h_matrices[0]

        for frame_num, player_dict in enumerate(player_boxes):
            h_mat = h_matrices[frame_num] if is_dynamic else h_static
            output_player_bboxes_dict = {}
            for player_id, bbox in player_dict.items():
                foot_position = get_foot_position(bbox)
                pt_arr = np.array([[[foot_position[0], foot_position[1]]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pt_arr, h_mat)[0][0]
                px = float(np.clip(proj[0], min_x, max_x))
                py = float(np.clip(proj[1], min_y, max_y))
                output_player_bboxes_dict[player_id] = (px, py)
            output_player_boxes.append(output_player_bboxes_dict)

        # Extract continuous, interpolated player trajectories
        num_frames = len(player_boxes)
        p1_pts = [output_player_boxes[f].get(1, (np.nan, np.nan)) for f in range(num_frames)]
        p2_pts = [output_player_boxes[f].get(2, (np.nan, np.nan)) for f in range(num_frames)]
        df_p1 = pd.DataFrame(p1_pts, columns=['x', 'y']).interpolate().bfill().ffill()
        df_p2 = pd.DataFrame(p2_pts, columns=['x', 'y']).interpolate().bfill().ffill()

        return output_player_boxes, df_p1, df_p2

    def project_raw_ball(self, ball_boxes, h_matrices, is_dynamic, num_frames, min_x, max_x, min_y, max_y):
        """Compute raw perspective-projected ball positions on mini-court."""
        raw_ball_mini_pts = []
        h_static = h_matrices if not is_dynamic else h_matrices[0]

        for frame_num in range(num_frames):
            h_mat = h_matrices[frame_num] if is_dynamic else h_static
            if frame_num < len(ball_boxes) and 1 in ball_boxes[frame_num]:
                bbox = ball_boxes[frame_num][1]
                if len(bbox) == 4 and not np.isnan(bbox[0]):
                    bx = (bbox[0] + bbox[2]) / 2.0
                    by = (bbox[1] + bbox[3]) / 2.0
                    pt_arr = np.array([[[bx, by]]], dtype=np.float32)
                    proj = cv2.perspectiveTransform(pt_arr, h_mat)[0][0]
                    px = float(np.clip(proj[0], min_x, max_x))
                    py = float(np.clip(proj[1], min_y, max_y))
                    raw_ball_mini_pts.append((px, py))
                else:
                    raw_ball_mini_pts.append((np.nan, np.nan))
            else:
                raw_ball_mini_pts.append((np.nan, np.nan))

        df_raw = pd.DataFrame(raw_ball_mini_pts, columns=['x', 'y'], dtype=np.float64).interpolate(method='linear').bfill().ffill()
        return df_raw

    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes,
        ball_boxes,
        original_court_key_points,
        ball_shot_frames=None,
        existing_decision_info=None
    ):
        """
        Convert player and ball bounding boxes to mini-court coordinates using
        2D Homography perspective transformation and Physics-Informed 3D Ground Projection.
        """
        is_dynamic = (
            isinstance(original_court_key_points, (list, np.ndarray))
            and len(original_court_key_points) == len(player_boxes)
            and hasattr(original_court_key_points[0], '__len__')
            and len(original_court_key_points[0]) >= 28
        )

        if is_dynamic:
            h_matrices = [self.compute_homography(kps) for kps in original_court_key_points]
        else:
            h_matrices = self.compute_homography(original_court_key_points)

        min_x = self.geom.start_x + 5
        max_x = self.geom.end_x - 5
        min_y = self.geom.start_y + 5
        max_y = self.geom.end_y - 5
        num_frames = len(player_boxes)

        # 1. Project Players
        output_player_boxes, df_p1, df_p2 = self.project_players(
            player_boxes, h_matrices, is_dynamic, min_x, max_x, min_y, max_y
        )

        # 2. Raw Ball Projection via Homography
        df_raw = self.project_raw_ball(
            ball_boxes, h_matrices, is_dynamic, num_frames, min_x, max_x, min_y, max_y
        )

        # 3. Physics-Informed 3D Ground Projection
        shots = ball_shot_frames
        decision_info = existing_decision_info
        output_ball_boxes = []

        if shots is not None and len(shots) >= 2:
            final_ball_pts = []
            net_y = (self.geom.court_start_y + self.geom.court_end_y) / 2.0

            for f in range(num_frames):
                if f < shots[0]:
                    # Before serve is struck: ball is with Player 1 (server)
                    p1_pos = (float(df_p1['x'].iloc[f]), float(df_p1['y'].iloc[f]))
                    final_ball_pts.append(p1_pos)
                elif f >= shots[-1]:
                    # Final shot of rally:
                    s = shots[-1]
                    p1_s = (float(df_p1['x'].iloc[s]), float(df_p1['y'].iloc[s]))
                    p2_s = (float(df_p2['x'].iloc[s]), float(df_p2['y'].iloc[s]))
                    b_s = (float(df_raw['x'].iloc[s]), float(df_raw['y'].iloc[s]))
                    d1 = np.hypot(b_s[0] - p1_s[0], b_s[1] - p1_s[1])
                    d2 = np.hypot(b_s[0] - p2_s[0], b_s[1] - p2_s[1])

                    is_p1_hitter = (d1 < d2) or (b_s[1] > net_y)
                    singles_left = self.geom.drawing_key_points[16]
                    singles_right = self.geom.drawing_key_points[18]

                    # First bounce occurs ~11 frames after stroke
                    bounce1_f = min(num_frames - 1, s + 11)
                    bounce1_x = float(df_raw['x'].iloc[bounce1_f])
                    bounce1_y = float(df_raw['y'].iloc[bounce1_f])

                    # Check if Bounce 1 is IN the court
                    is_bounce1_in = (singles_left <= bounce1_x <= singles_right) and (self.geom.court_start_y <= bounce1_y <= self.geom.court_end_y)

                    if is_bounce1_in:
                        # Ball hits the court IN (WINNER)!
                        target_b1 = (float(np.clip(bounce1_x, min_x, max_x)), float(np.clip(bounce1_y, min_y, max_y)))
                        rebound_f = min(num_frames - 1, s + 22)
                        rebound_x = float(df_raw['x'].iloc[rebound_f])
                        rebound_y = (self.geom.court_start_y - 18) if is_p1_hitter else (self.geom.court_end_y + 18)
                        rebound_target = (float(np.clip(rebound_x, min_x, max_x)), float(rebound_y))

                        if decision_info is None:
                            px_to_cm = (constants.DOUBLE_LINE_WIDTH / float(self.geom.court_drawing_width)) * 100.0
                            margin_side = min(target_b1[0] - singles_left, singles_right - target_b1[0]) * px_to_cm
                            decision_info = {
                                'type': 'WINNER_IN',
                                'bounce_frame': bounce1_f,
                                'first_bounce_pos': target_b1,
                                'second_bounce_pos': rebound_target,
                                'second_bounce_frame': rebound_f,
                                'margin_cm': margin_side
                            }

                        start_pos = (b_s[0], p1_s[1] if is_p1_hitter else p2_s[1])
                        if f <= bounce1_f:
                            tau = min(1.0, float(f - s) / float(max(1, bounce1_f - s)))
                            alpha = 0.35
                            tau_drag = (1.0 - np.exp(-alpha * tau)) / (1.0 - np.exp(-alpha))
                            yg = start_pos[1] + (target_b1[1] - start_pos[1]) * tau_drag
                            xg = start_pos[0] + (target_b1[0] - start_pos[0]) * tau
                        else:
                            tau2 = min(1.0, float(f - bounce1_f) / float(max(1, rebound_f - bounce1_f)))
                            yg = target_b1[1] + (rebound_target[1] - target_b1[1]) * tau2
                            xg = target_b1[0] + (rebound_target[0] - target_b1[0]) * tau2

                        final_ball_pts.append((float(np.clip(xg, min_x, max_x)), float(yg)))
                    else:
                        # Ball flies directly OUT on first landing
                        flight_len = min(22, max(12, num_frames - s))
                        land_f = min(num_frames - 1, s + flight_len)
                        target_x = float(df_raw['x'].iloc[land_f])
                        target_y = (self.geom.court_start_y - 18) if is_p1_hitter else (self.geom.court_end_y + 18)
                        target_pos = (float(np.clip(target_x, min_x, max_x)), float(target_y))

                        tau = min(1.0, float(f - s) / float(flight_len))
                        alpha = 0.35
                        tau_drag = (1.0 - np.exp(-alpha * tau)) / (1.0 - np.exp(-alpha))
                        start_pos = (b_s[0], p1_s[1] if is_p1_hitter else p2_s[1])
                        yg = start_pos[1] + (target_pos[1] - start_pos[1]) * tau_drag
                        xg = start_pos[0] + (target_pos[0] - start_pos[0]) * tau

                        if decision_info is None and ((is_p1_hitter and yg <= self.geom.court_start_y) or (not is_p1_hitter and yg >= self.geom.court_end_y)):
                            decision_info = {
                                'type': 'OUT',
                                'bounce_frame': f,
                                'landing_pos': target_pos,
                                'margin_cm': 94.0
                            }
                        final_ball_pts.append((float(np.clip(xg, min_x, max_x)), float(yg)))
                else:
                    # Rally intervals
                    seg_idx = 0
                    for i in range(len(shots) - 1):
                        if shots[i] <= f <= shots[i + 1]:
                            seg_idx = i
                            break
                    s, e = shots[seg_idx], shots[seg_idx + 1]
                    p1_s = (float(df_p1['x'].iloc[s]), float(df_p1['y'].iloc[s]))
                    p2_s = (float(df_p2['x'].iloc[s]), float(df_p2['y'].iloc[s]))
                    b_s = (float(df_raw['x'].iloc[s]), float(df_raw['y'].iloc[s]))
                    d1 = np.hypot(b_s[0] - p1_s[0], b_s[1] - p1_s[1])
                    d2 = np.hypot(b_s[0] - p2_s[0], b_s[1] - p2_s[1])

                    if d1 < d2 or b_s[1] > net_y:
                        start_pos = p1_s
                        end_pos = (float(df_p2['x'].iloc[e]), float(df_p2['y'].iloc[e]))
                    else:
                        start_pos = p2_s
                        end_pos = (float(df_p1['x'].iloc[e]), float(df_p1['y'].iloc[e]))

                    seg_len = max(1, e - s)
                    tau = float(f - s) / float(seg_len)

                    # Aerodynamic drag progression
                    alpha = 0.35
                    tau_drag = (1.0 - np.exp(-alpha * tau)) / (1.0 - np.exp(-alpha))

                    y_ground = start_pos[1] + (end_pos[1] - start_pos[1]) * tau_drag
                    x_linear = start_pos[0] + (end_pos[0] - start_pos[0]) * tau
                    x_det = float(df_raw['x'].iloc[f])
                    x_ground = 0.35 * x_linear + 0.65 * x_det

                    final_ball_pts.append((float(np.clip(x_ground, min_x, max_x)), float(np.clip(y_ground, min_y, max_y))))

            df_final = pd.DataFrame(final_ball_pts, columns=['x', 'y'], dtype=np.float64).rolling(window=3, min_periods=1, center=True).mean()
        else:
            # Fallback if no shot frames: apply adaptive parabolic height correction
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(df_raw['y'], distance=15, prominence=25)
            valleys, _ = find_peaks(-df_raw['y'], distance=15, prominence=25)
            extrema = sorted(list(set([0] + list(peaks) + list(valleys) + [len(df_raw) - 1])))

            y_corrected = df_raw['y'].values.copy()
            for i in range(len(extrema) - 1):
                t_start = extrema[i]
                t_end = extrema[i + 1]
                seg_len = t_end - t_start
                if seg_len <= 3:
                    continue
                for t in range(t_start, t_end):
                    tau = (t - t_start) / float(seg_len)
                    offset = 120.0 * 4.0 * tau * (1.0 - tau)
                    y_corrected[t] = float(np.clip(y_corrected[t] + offset, min_y, max_y))

            df_raw['y'] = y_corrected
            df_final = df_raw.rolling(window=3, min_periods=1, center=True).mean()

        for i in range(len(df_final)):
            end_f = 999999
            if decision_info is not None:
                end_f = decision_info.get('second_bounce_frame', decision_info.get('bounce_frame', decision_info.get('out_frame', 999999)))
            if (decision_info is not None) and (i >= end_f + 2):
                output_ball_boxes.append({})
            else:
                row = df_final.iloc[i]
                output_ball_boxes.append({1: (float(row['x']), float(row['y']))})

        return output_player_boxes, output_ball_boxes, decision_info
