import math
import numpy as np
import pandas as pd

class PlayerMetricsAnalyzer:
    """
    Trích xuất và tính toán các chỉ số di chuyển & thể lực của cầu thủ:
    - Quãng đường di chuyển tổng thể (m) với bộ lọc rung micro-jitter
    - Vận tốc chạy tức thời (km/h), vận tốc trung bình và bứt tốc tối đa
    - Tỷ lệ phân bố thời gian tại 3 vùng sân chiến thuật
    - Xu hướng di chuyển ngang (Ad Court vs Center vs Deuce Court)
    - Xây dựng bảng theo dõi Frame-by-Frame (player_tracking.csv)
    """
    def __init__(self, geometry):
        self.geom = geometry

    def compute_player_metrics(self, player_mini_court_detections, num_frames, video_fps, mini_court):
        pids = [1, 2]
        coords = {pid: [] for pid in pids}

        # 1. Trích xuất chuỗi tọa độ (m)
        for f in range(num_frames):
            frame_dict = player_mini_court_detections[f] if f < len(player_mini_court_detections) else {}
            for pid in pids:
                if pid in frame_dict and frame_dict[pid] is not None:
                    px, py = frame_dict[pid]
                    xm, ym = self.geom.mini_px_to_court_meters(mini_court, px, py)
                else:
                    xm, ym = np.nan, np.nan
                coords[pid].append((xm, ym))

        # 2. Nội suy tọa độ thiếu
        for pid in pids:
            df_c = pd.DataFrame(coords[pid], columns=['x_m', 'y_m']).interpolate().bfill().ffill()
            coords[pid] = list(zip(df_c['x_m'].values, df_c['y_m'].values))

        player_metrics = {}
        for pid in pids:
            c_list = coords[pid]
            distances = [0.0]
            speeds_kmh = [0.0]
            zones = []

            for i in range(1, num_frames):
                x0, y0 = c_list[i - 1]
                x1, y1 = c_list[i]
                d = math.hypot(x1 - x0, y1 - y0)
                # Lọc rung lắc sensor < 2cm mỗi frame
                if d < 0.02:
                    d = 0.0
                distances.append(d)
                v = (d * video_fps) * 3.6  # m/s -> km/h
                speeds_kmh.append(v)

            # Làm mịn đường cong vận tốc (rolling window 5 frames)
            speeds_smooth = pd.Series(speeds_kmh).rolling(window=5, min_periods=1, center=True).mean().values

            # Phân loại 3 vùng sân & xu hướng di chuyển ngang
            baseline_count = 0
            nomans_count = 0
            attack_count = 0
            lateral_deuce = 0
            lateral_center = 0
            lateral_ad = 0

            for i in range(num_frames):
                x, y = c_list[i]
                zone = self.geom.classify_zone(y)
                zones.append(zone)

                if "Baseline" in zone:
                    baseline_count += 1
                elif "No-Man's" in zone:
                    nomans_count += 1
                else:
                    attack_count += 1

                if x < -0.8:
                    lateral_ad += 1
                elif x > 0.8:
                    lateral_deuce += 1
                else:
                    lateral_center += 1

            total_dist = float(sum(distances))
            avg_speed = float(np.mean(speeds_smooth))
            max_speed = float(np.max(speeds_smooth))
            total_f = max(1, num_frames)

            player_metrics[pid] = {
                "player_id": pid,
                "player_name": f"Player {pid}" if pid == 1 else "Player 2",
                "court_side": "Near Court (San gan)" if pid == 1 else "Far Court (San xa)",
                "total_distance_m": round(total_dist, 2),
                "avg_speed_kmh": round(avg_speed, 1),
                "max_speed_kmh": round(max_speed, 1),
                "zone_baseline_pct": round((baseline_count / total_f) * 100.0, 1),
                "zone_nomans_land_pct": round((nomans_count / total_f) * 100.0, 1),
                "zone_attack_pct": round((attack_count / total_f) * 100.0, 1),
                "lateral_ad_pct": round((lateral_ad / total_f) * 100.0, 1),
                "lateral_center_pct": round((lateral_center / total_f) * 100.0, 1),
                "lateral_deuce_pct": round((lateral_deuce / total_f) * 100.0, 1),
                "speeds": speeds_smooth,
                "zones": zones,
                "coords": c_list
            }

        # 3. Tạo bảng Player Tracking Frame-by-Frame
        tracking_rows = []
        for f in range(num_frames):
            time_s = round(f / video_fps, 3)
            tracking_rows.append({
                "frame": f,
                "time_s": time_s,
                "p1_x_m": round(coords[1][f][0], 3),
                "p1_y_m": round(coords[1][f][1], 3),
                "p1_speed_kmh": round(float(player_metrics[1]["speeds"][f]), 2),
                "p1_zone": player_metrics[1]["zones"][f],
                "p2_x_m": round(coords[2][f][0], 3),
                "p2_y_m": round(coords[2][f][1], 3),
                "p2_speed_kmh": round(float(player_metrics[2]["speeds"][f]), 2),
                "p2_zone": player_metrics[2]["zones"][f],
            })

        df_player_tracking = pd.DataFrame(tracking_rows)
        return player_metrics, df_player_tracking
