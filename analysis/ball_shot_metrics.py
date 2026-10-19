import math
import numpy as np
import pandas as pd

class BallShotMetricsAnalyzer:
    """
    Trích xuất và phân tích chi tiết dữ liệu bóng và các cú đánh trong trận:
    - Bám vết bóng (Camera & Mini-Court) theo thời gian thực
    - Chuẩn hóa các sự kiện bóng nảy sân (Bounce Events & Hawk-Eye ELC)
    - Phân tích chi tiết từng cú đánh trong rally (Tốc độ, Hướng đánh, Độ sâu, Kết quả)
    - Tổng hợp từ điển chỉ số cấp cao (Summary KPIs)
    """
    def __init__(self, geometry):
        self.geom = geometry

    def compute_ball_tracking(self, ball_detections, ball_mini_court_detections, all_bounces, decision_info, num_frames, video_fps, mini_court):
        """Xử lý tọa độ bóng theo frame và chuẩn hóa danh sách điểm nảy sân."""
        ball_rows = []
        for f in range(num_frames):
            time_s = round(f / video_fps, 3)
            # Camera frame bbox
            cam_x, cam_y = np.nan, np.nan
            if f < len(ball_detections) and 1 in ball_detections[f]:
                box = ball_detections[f][1]
                if len(box) == 4 and not np.isnan(box[0]):
                    cam_x = (box[0] + box[2]) / 2.0
                    cam_y = (box[1] + box[3]) / 2.0

            # Mini-court coords in meters
            ball_x_m, ball_y_m = np.nan, np.nan
            if f < len(ball_mini_court_detections) and 1 in ball_mini_court_detections[f]:
                mpt = ball_mini_court_detections[f][1]
                if mpt is not None:
                    ball_x_m, ball_y_m = self.geom.mini_px_to_court_meters(mini_court, mpt[0], mpt[1])

            ball_rows.append({
                "frame": f,
                "time_s": time_s,
                "ball_cam_x": round(cam_x, 1) if not np.isnan(cam_x) else None,
                "ball_cam_y": round(cam_y, 1) if not np.isnan(cam_y) else None,
                "ball_x_m": round(ball_x_m, 3) if not np.isnan(ball_x_m) else None,
                "ball_y_m": round(ball_y_m, 3) if not np.isnan(ball_y_m) else None,
            })

        df_ball_tracking = pd.DataFrame(ball_rows)

        # Chuẩn hóa danh sách Bounce Events
        bounces_list = []
        b_source = all_bounces if all_bounces else []
        for idx, b in enumerate(b_source):
            f = int(b.get("frame", 0))
            is_in = bool(b.get("is_in", True))
            is_final = bool(b.get("is_final", False))
            shot_idx = int(b.get("shot_idx", idx))

            # Mini court position
            mini_pos = b.get("mini_pos")
            if mini_pos is not None:
                x_m, y_m = self.geom.mini_px_to_court_meters(mini_court, mini_pos[0], mini_pos[1])
            else:
                x_m, y_m = 0.0, 0.0

            court_side = "Far Court (Player 2)" if y_m < 0 else "Near Court (Player 1)"
            dist_to_sideline = abs(abs(x_m) - (self.geom.court_width_singles / 2.0))
            depth_cat = self.geom.classify_depth(y_m)

            margin_cm = float(b.get("margin_side", dist_to_sideline * 100.0))
            if is_final and decision_info is not None:
                margin_cm = float(decision_info.get("margin_cm", margin_cm))
                is_in = decision_info.get("decision") in ("IN", "WINNER")

            bounces_list.append({
                "bounce_index": idx + 1,
                "frame": f,
                "time_s": round(f / video_fps, 2),
                "shot_index": shot_idx + 1,
                "court_side": court_side,
                "x_m": round(x_m, 2),
                "y_m": round(y_m, 2),
                "is_in": is_in,
                "status": "IN" if is_in else "OUT",
                "depth_category": depth_cat,
                "margin_cm": round(margin_cm, 1),
                "is_final": is_final
            })

        return df_ball_tracking, bounces_list

    def compute_shot_details(
        self,
        ball_shot_frames,
        player_mini_court_detections,
        ball_mini_court_detections,
        bounces_list,
        player_stats_data_df,
        decision_info,
        num_frames,
        video_fps,
        mini_court
    ):
        """Phân tích chi tiết từng cú đánh trong rally."""
        shots = []
        if not ball_shot_frames:
            return pd.DataFrame()

        shot_frames_extended = list(ball_shot_frames)
        if shot_frames_extended and (num_frames - 1 - shot_frames_extended[-1] >= 8):
            shot_frames_extended.append(num_frames - 1)

        for s_idx in range(len(shot_frames_extended) - 1):
            start_f = shot_frames_extended[s_idx]
            end_f = shot_frames_extended[s_idx + 1]
            flight_frames = max(1, end_f - start_f)
            flight_time_s = round(flight_frames / video_fps, 2)

            # Xác định người đánh (Hitter)
            p_dict = player_mini_court_detections[start_f] if start_f < len(player_mini_court_detections) else {}
            b_pt = ball_mini_court_detections[start_f].get(1) if start_f < len(ball_mini_court_detections) else None

            if p_dict and b_pt:
                hitter_id = min(p_dict.keys(), key=lambda pid: math.hypot(p_dict[pid][0] - b_pt[0], p_dict[pid][1] - b_pt[1]))
            else:
                hitter_id = 1 if (s_idx % 2 == 0) else 2

            hitter_name = f"Player {hitter_id}"
            receiver_id = 2 if hitter_id == 1 else 1

            # Tọa độ người đánh (m)
            if p_dict and hitter_id in p_dict:
                hx_m, hy_m = self.geom.mini_px_to_court_meters(mini_court, p_dict[hitter_id][0], p_dict[hitter_id][1])
            else:
                hx_m, hy_m = 0.0, 11.0 if hitter_id == 1 else -11.0

            # Điểm nảy bóng tương ứng
            shot_bounces = [b for b in bounces_list if start_f <= b['frame'] <= end_f + 8]
            bounce_info = shot_bounces[0] if shot_bounces else None

            if bounce_info is not None:
                bx_m = bounce_info['x_m']
                by_m = bounce_info['y_m']
                depth = bounce_info['depth_category']
                is_in = bounce_info['is_in']
                margin = bounce_info['margin_cm']
                b_frame = bounce_info['frame']
                dt_to_bounce = max(0.18, (b_frame - start_f) / video_fps)
            else:
                bx_m = 0.0
                by_m = -hy_m * 0.8
                depth = "Medium (Giua san)"
                is_in = True
                margin = 35.0
                dt_to_bounce = max(0.35, flight_time_s)

            # Tính tốc độ bóng (km/h) từ dữ liệu pipeline hoặc mô hình động học quỹ đạo
            speed_kmh = None
            if player_stats_data_df is not None and not player_stats_data_df.empty:
                col_name = f"player_{hitter_id}_last_shot_speed"
                if col_name in player_stats_data_df.columns:
                    val = player_stats_data_df.loc[player_stats_data_df['frame_num'] == start_f, col_name]
                    if not val.empty and val.values[0] > 0:
                        speed_kmh = float(val.values[0])

            if speed_kmh is None or speed_kmh <= 0 or speed_kmh == 95.0:
                dist_m = math.hypot(bx_m - hx_m, by_m - hy_m)
                speed_kmh = (dist_m / dt_to_bounce) * 3.6
                speed_kmh = float(np.clip(speed_kmh, 52.0, 185.0))

            # Hướng đánh bóng
            if abs(bx_m) < 1.2:
                direction = "Center (Giua san)"
            elif (hx_m * bx_m < -0.8):
                direction = "Crosscourt (Cheo san)"
            elif abs(bx_m - hx_m) < 2.0:
                direction = "Down the Line (Doc day)"
            else:
                direction = "Wide Angle (Goc rong)"

            # Kết quả cú đánh
            is_last_shot = (s_idx == len(shot_frames_extended) - 2)
            if is_last_shot and decision_info is not None:
                dec = decision_info.get("decision", "IN")
                winner = decision_info.get("point_winner", hitter_id)
                if winner == hitter_id:
                    result = f"WINNER ({dec})"
                else:
                    result = f"ERROR ({dec})"
            else:
                result = "In Play (Rally)"

            shots.append({
                "shot_index": s_idx + 1,
                "hit_frame": start_f,
                "end_frame": end_f,
                "flight_time_s": flight_time_s,
                "hitter": hitter_name,
                "hitter_id": hitter_id,
                "receiver": f"Player {receiver_id}",
                "speed_kmh": round(speed_kmh, 1),
                "hit_x_m": round(hx_m, 2),
                "hit_y_m": round(hy_m, 2),
                "bounce_x_m": round(bx_m, 2),
                "bounce_y_m": round(by_m, 2),
                "direction": direction,
                "depth": depth,
                "status": "IN" if is_in else "OUT",
                "margin_cm": round(margin, 1),
                "result": result
            })

        return pd.DataFrame(shots)

    def compile_summary_kpis(self, match_name, player_metrics, shots_df, decision_info, bounces_list, num_frames, video_fps):
        """Tổng hợp toàn bộ chỉ số vào từ điển JSON chuẩn cho dashboard/web."""
        total_shots = len(shots_df)
        duration_s = round(num_frames / video_fps, 2)

        if not shots_df.empty:
            avg_shot_speed = round(float(shots_df["speed_kmh"].mean()), 1)
            max_shot_speed = round(float(shots_df["speed_kmh"].max()), 1)
            fastest_shot_row = shots_df.loc[shots_df["speed_kmh"].idxmax()]
            fastest_hitter = fastest_shot_row["hitter"]
        else:
            avg_shot_speed = 0.0
            max_shot_speed = 0.0
            fastest_hitter = "N/A"

        if decision_info is not None:
            winner_id = decision_info.get("point_winner", 1)
            winner_name = f"Player {winner_id}"
            decision_type = decision_info.get("decision", "IN")
            margin_cm = decision_info.get("margin_cm", 0.0)
            reason = decision_info.get("reason", "")
        else:
            winner_name = "Player 1"
            decision_type = "IN"
            margin_cm = 0.0
            reason = "Pha bóng hoàn thành"

        p1_shots = shots_df[shots_df["hitter_id"] == 1] if not shots_df.empty else pd.DataFrame()
        p2_shots = shots_df[shots_df["hitter_id"] == 2] if not shots_df.empty else pd.DataFrame()

        p1_deep_ratio = round((len(p1_shots[p1_shots["depth"].str.contains("Deep", na=False)]) / max(1, len(p1_shots))) * 100.0, 1) if not p1_shots.empty else 0.0
        p2_deep_ratio = round((len(p2_shots[p2_shots["depth"].str.contains("Deep", na=False)]) / max(1, len(p2_shots))) * 100.0, 1) if not p2_shots.empty else 0.0

        summary = {
            "match_name": match_name,
            "total_frames": num_frames,
            "fps": video_fps,
            "duration_seconds": duration_s,
            "rally_length_shots": total_shots,
            "point_winner": winner_name,
            "decision": decision_type,
            "margin_cm": margin_cm,
            "referee_reason": reason,
            "ball_speed_kpis": {
                "avg_speed_kmh": avg_shot_speed,
                "max_speed_kmh": max_shot_speed,
                "fastest_hitter": fastest_hitter
            },
            "player_1": {
                "name": "Player 1",
                "court_side": "Near Court (San gan)",
                "total_distance_m": player_metrics[1]["total_distance_m"],
                "avg_speed_kmh": player_metrics[1]["avg_speed_kmh"],
                "max_speed_kmh": player_metrics[1]["max_speed_kmh"],
                "shots_hit": len(p1_shots),
                "deep_ball_ratio_pct": p1_deep_ratio,
                "zone_distribution_pct": {
                    "baseline": player_metrics[1]["zone_baseline_pct"],
                    "nomans_land": player_metrics[1]["zone_nomans_land_pct"],
                    "attack_zone": player_metrics[1]["zone_attack_pct"]
                },
                "lateral_distribution_pct": {
                    "ad_court": player_metrics[1]["lateral_ad_pct"],
                    "center": player_metrics[1]["lateral_center_pct"],
                    "deuce_court": player_metrics[1]["lateral_deuce_pct"]
                }
            },
            "player_2": {
                "name": "Player 2",
                "court_side": "Far Court (San xa)",
                "total_distance_m": player_metrics[2]["total_distance_m"],
                "avg_speed_kmh": player_metrics[2]["avg_speed_kmh"],
                "max_speed_kmh": player_metrics[2]["max_speed_kmh"],
                "shots_hit": len(p2_shots),
                "deep_ball_ratio_pct": p2_deep_ratio,
                "zone_distribution_pct": {
                    "baseline": player_metrics[2]["zone_baseline_pct"],
                    "nomans_land": player_metrics[2]["zone_nomans_land_pct"],
                    "attack_zone": player_metrics[2]["zone_attack_pct"]
                },
                "lateral_distribution_pct": {
                    "ad_court": player_metrics[2]["lateral_ad_pct"],
                    "center": player_metrics[2]["lateral_center_pct"],
                    "deuce_court": player_metrics[2]["lateral_deuce_pct"]
                }
            }
        }
        return summary
