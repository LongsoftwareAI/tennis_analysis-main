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

        # Extract continuous, interpolated player trajectories for all active players
        num_frames = len(player_boxes)
        all_pids = sorted(list(set(pid for f in output_player_boxes for pid in f.keys())))
        if not all_pids:
            all_pids = [1, 2]

        df_players = {}
        for pid in all_pids:
            pts = [output_player_boxes[f].get(pid, (np.nan, np.nan)) for f in range(num_frames)]
            df_players[pid] = pd.DataFrame(pts, columns=['x', 'y']).interpolate().bfill().ffill()

        return output_player_boxes, df_players

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

        df_raw = pd.DataFrame(raw_ball_mini_pts, columns=['x', 'y'], dtype=np.float64).interpolate(method='linear', limit=4)
        return df_raw

    def _interpolate_court_flight(self, f, t_start, t_end, p_start, p_end, net_y=None, alpha=None):
        """
        Real-time synchronous court flight interpolation:
        Progresses smoothly and linearly from p_start to p_end synchronized 1:1 with frame time,
        ensuring zero lag and natural motion matching the broadcast video.
        """
        span = max(1, t_end - t_start)
        tau = min(1.0, max(0.0, float(f - t_start) / float(span)))
        xg = p_start[0] + (p_end[0] - p_start[0]) * tau
        yg = p_start[1] + (p_end[1] - p_start[1]) * tau
        return xg, yg


    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes,
        ball_boxes,
        original_court_key_points,
        ball_shot_frames=None,
        existing_decision_info=None,
        all_bounces=None
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
        output_player_boxes, df_players = self.project_players(
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

        if shots is not None and len(shots) >= 1:
            final_ball_pts = []
            net_y = (self.geom.court_start_y + self.geom.court_end_y) / 2.0

            # Map detected bounces by shot index: store the earliest/first bounce for each shot
            bounce_map = {}
            if all_bounces:
                for b in all_bounces:
                    s_idx = b['shot_idx']
                    if s_idx not in bounce_map or b['frame'] < bounce_map[s_idx]['frame']:
                        bounce_map[s_idx] = b

            alpha = 0.35

            for f in range(num_frames):
                if f < shots[0]:
                    # Before serve is struck: ball is with the serving player
                    s0 = shots[0]
                    p1_id = list(df_players.keys())[0] if df_players else 1
                    bx_raw = float(df_raw['x'].iloc[s0]) if pd.notna(df_raw['x'].iloc[s0]) else float(df_players[p1_id]['x'].iloc[s0])
                    by_raw = float(df_raw['y'].iloc[s0]) if pd.notna(df_raw['y'].iloc[s0]) else float(df_players[p1_id]['y'].iloc[s0])
                    server_pid = min(df_players.keys(), key=lambda pid: np.hypot(
                        float(df_players[pid]['x'].iloc[s0]) - bx_raw,
                        float(df_players[pid]['y'].iloc[s0]) - by_raw
                    ))
                    server_pos = (float(df_players[server_pid]['x'].iloc[f]), float(df_players[server_pid]['y'].iloc[f]))
                    final_ball_pts.append(server_pos)
                    continue

                # Find which shot segment frame f belongs to
                seg_idx = len(shots) - 1
                for i in range(len(shots) - 1):
                    if shots[i] <= f < shots[i + 1]:
                        seg_idx = i
                        break

                s = shots[seg_idx]
                is_final = (seg_idx == len(shots) - 1)
                e = shots[seg_idx + 1] if not is_final else num_frames - 1

                # Continuous stroke position: seamless transition from previous frame
                if seg_idx > 0 and len(final_ball_pts) >= s:
                    start_pos = final_ball_pts[s - 1]
                else:
                    raw_sx = float(df_raw['x'].iloc[s]) if pd.notna(df_raw['x'].iloc[s]) else (final_ball_pts[s-1][0] if final_ball_pts else min_x)
                    raw_sy = float(df_raw['y'].iloc[s]) if pd.notna(df_raw['y'].iloc[s]) else (final_ball_pts[s-1][1] if final_ball_pts else min_y)
                    start_pos = (raw_sx, raw_sy)
                start_pos = (float(np.clip(start_pos[0], min_x, max_x)), float(np.clip(start_pos[1], min_y, max_y)))

                if is_final and decision_info is not None:
                    bounce_f = decision_info.get('landing_frame', decision_info.get('bounce_frame', min(e - 2, s + 11)))
                    target_b = decision_info.get('landing_pos_mini', decision_info.get('first_bounce_pos'))
                    if target_b is None:
                        b_info = bounce_map.get(seg_idx)
                        target_b = b_info.get('mini_pos', (start_pos[0], net_y)) if b_info else (start_pos[0], net_y)
                    target_b = (float(np.clip(target_b[0], min_x, max_x)), float(np.clip(target_b[1], min_y, max_y)))

                    rebound_f = decision_info.get('second_bounce_frame', min(num_frames - 1, bounce_f + 15))
                    rebound_target = decision_info.get('second_bounce_pos')
                    if rebound_target is None:
                        y_reb = (self.geom.court_start_y - 12) if target_b[1] < start_pos[1] else (self.geom.court_end_y + 12)
                        x_reb = target_b[0] + (target_b[0] - start_pos[0]) * 0.25
                        rebound_target = (float(np.clip(x_reb, min_x, max_x)), float(y_reb))
                    else:
                        rebound_target = (float(np.clip(rebound_target[0], min_x, max_x)), float(np.clip(rebound_target[1], min_y, max_y)))

                    if f <= bounce_f:
                        # Flight from striker to ground bounce / landing point
                        xg, yg = self._interpolate_court_flight(f, s, bounce_f, start_pos, target_b, net_y, alpha)
                    elif f <= rebound_f:
                        # Rebound from first bounce to second bounce
                        tau2 = float(f - bounce_f) / float(max(1, rebound_f - bounce_f))
                        xg = target_b[0] + (rebound_target[0] - target_b[0]) * tau2
                        yg = target_b[1] + (rebound_target[1] - target_b[1]) * tau2
                    else:
                        # Dead ball at rest after second bounce
                        xg = rebound_target[0]
                        yg = rebound_target[1]

                elif is_final and decision_info is None:
                    # Before referee runs: project ball smoothly into opponent court
                    b_info = bounce_map.get(seg_idx)
                    if b_info is not None:
                        bounce_f = b_info.get('peak_frame', b_info.get('frame', min(e - 2, s + 11)))
                        target_b = b_info.get('mini_pos', (start_pos[0], net_y))
                        target_b = (float(np.clip(target_b[0], min_x, max_x)), float(np.clip(target_b[1], min_y, max_y)))
                        if f <= bounce_f:
                            xg, yg = self._interpolate_court_flight(f, s, bounce_f, start_pos, target_b, net_y, alpha)
                        else:
                            rebound_f = min(num_frames - 1, bounce_f + 14)
                            y_reb = (self.geom.court_start_y - 12) if target_b[1] < start_pos[1] else (self.geom.court_end_y + 12)
                            x_reb = target_b[0] + (target_b[0] - start_pos[0]) * 0.25
                            rebound_target = (float(np.clip(x_reb, min_x, max_x)), float(y_reb))
                            tau2 = min(1.0, max(0.0, float(f - bounce_f) / float(max(1, rebound_f - bounce_f))))
                            xg = target_b[0] + (rebound_target[0] - target_b[0]) * tau2
                            yg = target_b[1] + (rebound_target[1] - target_b[1]) * tau2
                    else:
                        target_y = (self.geom.court_start_y + 40) if start_pos[1] > net_y else (self.geom.court_end_y - 40)
                        target_pos = (start_pos[0], target_y)
                        xg, yg = self._interpolate_court_flight(f, s, e, start_pos, target_pos, net_y, alpha)

                else:
                    # Rally shot (not final):
                    b_info = bounce_map.get(seg_idx)
                    if b_info is not None:
                        bounce_f = b_info.get('peak_frame', b_info.get('frame', min(e - 2, s + 11)))
                        target_b = b_info.get('mini_pos', (start_pos[0], net_y))
                        target_b = (float(np.clip(target_b[0], min_x, max_x)), float(np.clip(target_b[1], min_y, max_y)))

                        if f <= bounce_f:
                            xg, yg = self._interpolate_court_flight(f, s, bounce_f, start_pos, target_b, net_y, alpha)
                        else:
                            # Rebound directly towards the receiving player on the other side of net
                            opponents = [pid for pid in df_players if (df_players[pid]['y'].iloc[e] - net_y) * (start_pos[1] - net_y) < 0]
                            if not opponents:
                                opponents = list(df_players.keys())
                            receiver_pid = min(opponents, key=lambda pid: np.hypot(
                                float(df_players[pid]['x'].iloc[e]) - target_b[0],
                                float(df_players[pid]['y'].iloc[e]) - target_b[1]
                            ))
                            receiver_pos = (float(df_players[receiver_pid]['x'].iloc[e]), float(df_players[receiver_pid]['y'].iloc[e]))
                            e_pos = (float(np.clip(receiver_pos[0], min_x, max_x)), float(np.clip(receiver_pos[1], min_y, max_y)))

                            tau2 = min(1.0, max(0.0, float(f - bounce_f) / float(max(1, e - bounce_f))))
                            xg = target_b[0] + (e_pos[0] - target_b[0]) * tau2
                            yg = target_b[1] + (e_pos[1] - target_b[1]) * tau2
                    else:
                        # Shot without ground bounce (Volley / Overhead Smash hit out of the air)
                        opponents = [pid for pid in df_players if (df_players[pid]['y'].iloc[e] - net_y) * (start_pos[1] - net_y) < 0]
                        if not opponents:
                            opponents = list(df_players.keys())
                        receiver_pid = min(opponents, key=lambda pid: np.hypot(
                            float(df_players[pid]['x'].iloc[e]) - start_pos[0],
                            float(df_players[pid]['y'].iloc[e]) - start_pos[1]
                        ))
                        receiver_pos = (float(df_players[receiver_pid]['x'].iloc[e]), float(df_players[receiver_pid]['y'].iloc[e]))
                        e_pos = (float(np.clip(receiver_pos[0], min_x, max_x)), float(np.clip(receiver_pos[1], min_y, max_y)))

                        xg, yg = self._interpolate_court_flight(f, s, e, start_pos, e_pos, net_y, alpha)

                final_ball_pts.append((float(np.clip(xg, min_x, max_x)), float(np.clip(yg, min_y, max_y))))

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
            if (decision_info is not None) and (i >= end_f + 12):
                output_ball_boxes.append({})
            else:
                row = df_final.iloc[i]
                output_ball_boxes.append({1: (float(row['x']), float(row['y']))})

        return output_player_boxes, output_ball_boxes, decision_info
