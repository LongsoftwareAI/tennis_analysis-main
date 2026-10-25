import math
import numpy as np
import pandas as pd


class PlayerMetricsAnalyzer:
    """
    Trich xuat va tinh toan cac chi so di chuyen & the luc cua cau thu:
    - Quang duong di chuyen (m): loai bo diem "teleport" (ID switch / homography giat), median filter, deadband
    - Van toc chay tuc thoi (km/h) bi chan o nguong sinh ly, van toc trung binh va but toc toi da
    - Ty le phan bo thoi gian tai 3 vung san chien thuat
    - Xu huong di chuyen ngang (Ad Court vs Center vs Deuce Court)
    - Bang theo doi Frame-by-Frame (player_tracking.csv)
    """
    MAX_PLAYER_KMH = 36.0       # tran van toc hop ly cua van dong vien tennis (khop MAX_PLAYER_KMH o UI)
    STEP_TOLERANCE_M = 0.25     # du dia cho nhieu toa do khi kiem tra buoc nhay
    MAX_GAP_SEC = 2.0           # mat track qua lau -> chap nhan vi tri moi (tranh ket o diem cu)
    DEADBAND_M = 0.02           # buoc < 2cm/frame coi la dung yen

    def __init__(self, geometry):
        self.geom = geometry

    # ------------------------------------------------------------------
    def _reject_spikes(self, xy, fps):
        """
        Loai cac diem nhay phi thuc te. So sanh moi diem voi DIEM HOP LE GAN NHAT:
        neu khoang cach > van_toc_toi_da * so_frame_cach + dung_sai thi bo (thanh NaN).
        Cach nay xu ly duoc ca kieu "nhay sang ben kia san roi nhay ve" (spike 1-N frame).
        Tra ve (xy_da_loc, so_diem_bi_loai).
        """
        n = len(xy)
        out = np.full_like(xy, np.nan, dtype=float)
        max_step = self.MAX_PLAYER_KMH / 3.6 / fps
        max_gap = int(self.MAX_GAP_SEC * fps)
        last_i, last_p, rejected = None, None, 0

        for i in range(n):
            p = xy[i]
            if np.isnan(p).any():
                continue
            if last_p is None:
                out[i] = p
                last_i, last_p = i, p
                continue
            gap = i - last_i
            d = math.hypot(p[0] - last_p[0], p[1] - last_p[1])
            if d <= max_step * gap + self.STEP_TOLERANCE_M or gap > max_gap:
                out[i] = p
                last_i, last_p = i, p
            else:
                rejected += 1
        return out, rejected

    def _clean_track(self, raw_xy, fps):
        """raw_xy: (n,2) co NaN -> (n,2) muot, khong NaN (neu co du lieu), so diem bi loai."""
        xy, rejected = self._reject_spikes(raw_xy, fps)
        df = pd.DataFrame(xy, columns=["x", "y"])
        df = df.interpolate(limit_direction="both")
        # median 5 frame khu nhieu xung, roi mean 3 frame lam tron nhe
        df = df.rolling(5, center=True, min_periods=1).median()
        df = df.rolling(3, center=True, min_periods=1).mean()
        return df.to_numpy(dtype=float), rejected

    # ------------------------------------------------------------------
    def compute_player_metrics(self, player_mini_court_detections, num_frames, video_fps, mini_court):
        fps = float(video_fps) if video_fps and video_fps > 0 else 24.0

        # Tu dong lay tat ca PID xuat hien, du phong it nhat 1, 2
        seen = {pid for fr in player_mini_court_detections for pid in fr}
        pids = sorted({1, 2} | seen)

        # 1. Trich xuat chuoi toa do (m); NaN khi khong co du lieu
        raw = {pid: np.full((num_frames, 2), np.nan) for pid in pids}
        for f in range(num_frames):
            frame_dict = player_mini_court_detections[f] if f < len(player_mini_court_detections) else {}
            for pid in pids:
                pos = frame_dict.get(pid)
                if pos is not None:
                    raw[pid][f] = self.geom.mini_px_to_court_meters(
                        mini_court, pos[0], pos[1], nan_if_invalid=True)

        player_metrics = {}
        coords = {}
        for pid in pids:
            has_data = not np.isnan(raw[pid]).all()
            if has_data:
                xy, rejected = self._clean_track(raw[pid], fps)
            else:
                xy, rejected = raw[pid], 0
            coords[pid] = xy

            # 2. Quang duong & van toc tren toa do da lam sach
            if has_data and num_frames > 1:
                steps = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))
                steps[steps < self.DEADBAND_M] = 0.0
                steps = np.minimum(steps, self.MAX_PLAYER_KMH / 3.6 / fps)   # chan cung theo gioi han sinh ly
                distances = np.concatenate([[0.0], steps])
            else:
                distances = np.zeros(num_frames)

            speeds_kmh = distances * fps * 3.6
            speeds_smooth = pd.Series(speeds_kmh).rolling(window=5, min_periods=1, center=True).mean().values

            # 3. Phan loai vung san & xu huong ngang
            zones = []
            baseline_count = nomans_count = attack_count = 0
            lateral_deuce = lateral_center = lateral_ad = 0
            valid_frames = 0

            for i in range(num_frames):
                x, y = xy[i]
                if np.isnan(x) or np.isnan(y):
                    zones.append("N/A")
                    continue
                valid_frames += 1
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

            total_f = max(1, valid_frames)
            total_dist = float(distances.sum())
            avg_speed = float(np.mean(speeds_smooth)) if num_frames else 0.0
            max_speed = float(np.max(speeds_smooth)) if num_frames else 0.0
            is_near = (pid % 2 == 1)     # P1,P3 = doi gan camera ; P2,P4 = doi xa

            player_metrics[pid] = {
                "player_id": pid,
                "player_name": f"Player {pid}",
                "court_side": "Near Court (San gan)" if is_near else "Far Court (San xa)",
                "total_distance_m": round(total_dist, 2),
                "avg_speed_kmh": round(avg_speed, 1),
                "max_speed_kmh": round(max_speed, 1),
                "zone_baseline_pct": round((baseline_count / total_f) * 100.0, 1),
                "zone_nomans_land_pct": round((nomans_count / total_f) * 100.0, 1),
                "zone_attack_pct": round((attack_count / total_f) * 100.0, 1),
                "lateral_ad_pct": round((lateral_ad / total_f) * 100.0, 1),
                "lateral_center_pct": round((lateral_center / total_f) * 100.0, 1),
                "lateral_deuce_pct": round((lateral_deuce / total_f) * 100.0, 1),
                "rejected_jump_points": int(rejected),   # >0 nhieu => kiem tra ID switch / court keypoints
                "speeds": speeds_smooth,
                "zones": zones,
                "coords": [tuple(p) for p in xy],
            }
            if rejected:
                print(f"[PlayerMetrics] P{pid}: loai {rejected} diem nhay phi thuc te "
                      f"(> {self.MAX_PLAYER_KMH:.0f} km/h) khoi quang duong/van toc.")

        # 4. Bang Player Tracking Frame-by-Frame
        tracking_rows = []
        for f in range(num_frames):
            row = {"frame": f, "time_s": round(f / fps, 3)}
            for pid in pids:
                x, y = coords[pid][f]
                row[f"p{pid}_x_m"] = None if np.isnan(x) else round(float(x), 3)
                row[f"p{pid}_y_m"] = None if np.isnan(y) else round(float(y), 3)
                row[f"p{pid}_speed_kmh"] = round(float(player_metrics[pid]["speeds"][f]), 2)
                row[f"p{pid}_zone"] = player_metrics[pid]["zones"][f]
            tracking_rows.append(row)

        df_player_tracking = pd.DataFrame(tracking_rows)
        return player_metrics, df_player_tracking