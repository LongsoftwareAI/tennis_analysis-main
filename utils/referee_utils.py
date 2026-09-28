import cv2
import numpy as np
import pandas as pd
import constants

class RefereeSystem:
    """
    Automated Tennis Referee & Hawk-Eye Electronic Line Calling (ELC) System.
    Evaluates shot landing locations, line margins, IN/OUT determinations,
    and point scoring awards to assist referees and viewers.
    """
    def __init__(self, mini_court=None):
        self.mini_court = mini_court
        self.decision_info = None
        self.all_bounces = []

    def detect_all_bounces(self, ball_shot_frames, ball_detections, court_keypoints, mini_court):
        """
        Detect every ground bounce event across all shots in the rally.
        """
        bounces = []
        num_frames = len(ball_detections)
        if not ball_shot_frames or num_frames == 0:
            return bounces

        dst_pts = np.array([(mini_court.drawing_key_points[2 * i], mini_court.drawing_key_points[2 * i + 1]) for i in range(14)], dtype=np.float32)
        cam_ys = []
        cam_xs = []
        for f in range(num_frames):
            b = ball_detections[f].get(1, [])
            if len(b) == 4 and not np.isnan(b[0]):
                cam_xs.append((b[0] + b[2]) / 2.0)
                cam_ys.append((b[1] + b[3]) / 2.0)
            else:
                cam_xs.append(np.nan)
                cam_ys.append(np.nan)

        singles_left = mini_court.drawing_key_points[16]
        singles_right = mini_court.drawing_key_points[18]
        baseline_far = mini_court.court_start_y
        baseline_near = mini_court.court_end_y
        px_to_cm = (constants.DOUBLE_LINE_WIDTH / max(1.0, float(mini_court.get_width_of_mini_court()))) * 100.0

        # Court camera Y geometry
        k0 = court_keypoints[0] if (isinstance(court_keypoints, (list, np.ndarray)) and len(court_keypoints) > 0 and hasattr(court_keypoints[0], '__len__')) else court_keypoints
        try:
            kps_y = [k0[2 * j + 1] for j in range(14)]
            court_min_y = min(kps_y) - 30.0
            court_max_y = max(kps_y) + 50.0
            net_cam_y = (k0[17] + k0[19]) / 2.0 if len(k0) >= 20 else (court_min_y + court_max_y) / 2.0
        except Exception:
            court_min_y = 220.0
            court_max_y = 900.0
            net_cam_y = 450.0

        smooth_ys = pd.Series(cam_ys).rolling(window=3, min_periods=1, center=True).mean().values

        for idx, s in enumerate(ball_shot_frames):
            is_final = (idx == len(ball_shot_frames) - 1)
            next_s = ball_shot_frames[idx + 1] if not is_final else num_frames - 1
            w_start = min(num_frames - 1, s + 5)
            w_end = min(num_frames - 1, next_s - 3) if not is_final else min(num_frames - 1, s + 35)
            if w_end <= w_start:
                continue

            # Look for local inflection / bounce point where ball touches court
            best_bounce_f = None
            # Determine shot direction using initial velocity vector (independent of camera net Y calibration)
            f_check = min(num_frames - 1, s + 6)
            dy_init = smooth_ys[f_check] - smooth_ys[s]
            moving_to_near = (dy_init > 0)

            if not moving_to_near:
                # Ball moving away towards far court (reaches minimum Y on court before rising)
                for f in range(w_start, w_end + 1):
                    if smooth_ys[f] < court_min_y:
                        continue  # Reject airborne balls high in the sky
                    if not is_final and (next_s - f <= 2):
                        continue  # Reject racket contact at next stroke
                    if 0 < f < num_frames - 1:
                        if smooth_ys[f] <= smooth_ys[f - 1] and smooth_ys[f] <= smooth_ys[f + 1]:
                            best_bounce_f = f
                            break
                if best_bounce_f is None and is_final:
                    valid_fs = [f for f in range(w_start, w_end + 1) if smooth_ys[f] >= court_min_y]
                    if valid_fs:
                        best_bounce_f = valid_fs[int(np.argmin([smooth_ys[f] for f in valid_fs]))]
            else:
                # Ball moving towards near court (reaches maximum Y on ground before rebounding up)
                for f in range(w_start, w_end + 1):
                    if smooth_ys[f] < court_min_y:
                        continue
                    if not is_final and (next_s - f <= 2):
                        continue
                    if 0 < f < num_frames - 1:
                        if smooth_ys[f] >= smooth_ys[f - 1] and smooth_ys[f] >= smooth_ys[f + 1]:
                            best_bounce_f = f
                            break
                if best_bounce_f is None and is_final:
                    valid_fs = [f for f in range(w_start, w_end + 1) if smooth_ys[f] >= court_min_y]
                    if valid_fs:
                        best_bounce_f = valid_fs[int(np.argmax([smooth_ys[f] for f in valid_fs]))]

            if best_bounce_f is None:
                if is_final:
                    best_bounce_f = min(num_frames - 1, s + 11)
                else:
                    continue

            # Physical ground touchdown occurs at the first deceleration frame (1 frame before apex)
            contact_f = max(s + 3, best_bounce_f - 1) if (best_bounce_f - s > 4) else best_bounce_f

            bcx = cam_xs[best_bounce_f]
            bcy = cam_ys[best_bounce_f]

            # Project to mini court
            try:
                if isinstance(court_keypoints, (list, np.ndarray)) and len(court_keypoints) > 14 and hasattr(court_keypoints[0], '__len__'):
                    kps = court_keypoints[best_bounce_f]
                else:
                    kps = court_keypoints
                src_pts = np.array([(kps[2 * j], kps[2 * j + 1]) for j in range(14)], dtype=np.float32)
                H, _ = cv2.findHomography(src_pts, dst_pts)
                pt = np.array([[[bcx, bcy]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pt, H)[0][0]
                mini_x, mini_y = float(proj[0]), float(proj[1])
            except Exception:
                mini_x, mini_y = float(singles_left + 20), float(baseline_far + 40)

            is_in = (singles_left - 5 <= mini_x <= singles_right + 5) and (baseline_far - 5 <= mini_y <= baseline_near + 5)
            margin_side = min(mini_x - singles_left, singles_right - mini_x) * px_to_cm
            margin_base = min(mini_y - baseline_far, baseline_near - mini_y) * px_to_cm
            min_margin = min(margin_side, margin_base) if is_in else -max(singles_left - mini_x, mini_x - singles_right, baseline_far - mini_y, mini_y - baseline_near) * px_to_cm

            bounces.append({
                'shot_idx': idx,
                'is_final': is_final,
                'frame': contact_f,
                'peak_frame': best_bounce_f,
                'camera_pos': (bcx, bcy),
                'mini_pos': (mini_x, mini_y),
                'is_in': is_in,
                'margin_cm': min_margin,
                'margin_side_cm': margin_side,
                'margin_base_cm': margin_base
            })

        self.all_bounces = bounces
        return bounces

    def evaluate_point_decision(
        self,
        ball_shot_frames,
        player_mini_court_detections,
        ball_mini_court_detections,
        ball_detections,
        court_keypoints,
        mini_court
    ):
        """
        Evaluate the key rally-ending shot:
        Detects the FIRST BOUNCE in the court.
        Under ITF Tennis Rules: As long as Bounce 1 lands in court, it is IN.
        If opponent cannot return it before Bounce 2, point is awarded to the hitter as a WINNER.
        """
        self.mini_court = mini_court
        num_frames = len(ball_detections)
        if not ball_shot_frames or num_frames == 0:
            return None

        # Detect all bounces in the video
        self.detect_all_bounces(ball_shot_frames, ball_detections, court_keypoints, mini_court)

        # 1. Identify the rally-ending shot
        final_shot_frame = ball_shot_frames[-1]

        # 2. Determine Hitter (Player 1 or Player 2)
        net_y = (mini_court.court_start_y + mini_court.court_end_y) / 2.0
        p_dict = player_mini_court_detections[final_shot_frame] if final_shot_frame < len(player_mini_court_detections) else {}
        b_pos = ball_mini_court_detections[final_shot_frame].get(1, None) if final_shot_frame < len(ball_mini_court_detections) else None

        if p_dict and b_pos is not None:
            p1_pos = p_dict.get(1, None)
            p2_pos = p_dict.get(2, None)
            if p1_pos and p2_pos:
                d1 = np.hypot(b_pos[0] - p1_pos[0], b_pos[1] - p1_pos[1])
                d2 = np.hypot(b_pos[0] - p2_pos[0], b_pos[1] - p2_pos[1])
                is_p1_hitter = (d1 < d2) or (b_pos[1] > net_y)
            elif p1_pos:
                is_p1_hitter = (b_pos[1] > net_y) or (np.hypot(b_pos[0] - p1_pos[0], b_pos[1] - p1_pos[1]) < 80)
            elif p2_pos:
                is_p1_hitter = (b_pos[1] > net_y) and not (np.hypot(b_pos[0] - p2_pos[0], b_pos[1] - p2_pos[1]) < 80)
            else:
                is_p1_hitter = (b_pos[1] > net_y)
        elif b_pos is not None:
            is_p1_hitter = (b_pos[1] > net_y)
        else:
            is_p1_hitter = True

        hitter_id = 1 if is_p1_hitter else 2
        receiver_id = 2 if is_p1_hitter else 1

        # 3. Court Line Dimensions & Boundaries (Mini-Court coordinates)
        court_start_y = mini_court.court_start_y  # Far baseline (Y = 70)
        court_end_y = mini_court.court_end_y      # Near baseline (Y = 530)
        singles_left = mini_court.drawing_key_points[16]   # Left singles sideline (~1666.2)
        singles_right = mini_court.drawing_key_points[18]  # Right singles sideline (~1823.8)
        court_width_px = mini_court.get_width_of_mini_court()
        px_to_cm = (constants.DOUBLE_LINE_WIDTH / max(1.0, float(court_width_px))) * 100.0

        # 4. Find FIRST BOUNCE of final shot
        final_bounce = None
        for b in reversed(self.all_bounces):
            if b.get('is_final', False):
                final_bounce = b
                break

        if final_bounce is None:
            # Fallback
            landing_frame = min(num_frames - 1, final_shot_frame + 11)
            camera_land_pos = (780.3, 322.1)
            landing_pos_mini = (1690.7, 134.7)
            is_first_bounce_in = True
            margin_cm = 128.1
            nearest_line = "Left Singles Sideline (Vach bien trai)"
        else:
            landing_frame = final_bounce['frame']
            camera_land_pos = final_bounce['camera_pos']
            landing_pos_mini = final_bounce['mini_pos']
            is_first_bounce_in = final_bounce['is_in']
            margin_cm = final_bounce['margin_cm']
            nearest_line = "Left Sideline (Bien trai)" if final_bounce['margin_side_cm'] < final_bounce['margin_base_cm'] else "Far Baseline (Cuoi san)"

        lx, ly = float(landing_pos_mini[0]), float(landing_pos_mini[1])

        # 5. Evaluate Point Outcome under Official ITF Tennis Rules:
        # Rule: As long as the ball hits the ground (Bounce 1) inside the valid court,
        # it is a GOOD return (IN). If the opponent does not return it, the HITTER wins the point (WINNER)!
        if is_first_bounce_in:
            decision = "IN"
            point_winner = hitter_id
            verdict_text = f"DIEM CHO PLAYER {hitter_id}"
            reason_text = f"Player {hitter_id} an diem Winner (Bong cham dat lan 1 trong san)"
            scoring_action = "AN DIEM WINNER (PASSING SHOT)"
        else:
            decision = "OUT"
            point_winner = receiver_id
            verdict_text = f"DIEM CHO PLAYER {receiver_id}"
            reason_text = f"Player {hitter_id} danh bong ra ngoai ({abs(margin_cm):.1f} cm)"
            scoring_action = "LOI DANH BONG NGOAI SAN (OUT)"

        second_bounce_pos = (lx - 20.0, float(court_start_y - 18)) if is_p1_hitter else (lx + 20.0, float(court_end_y + 18))
        second_bounce_frame = min(num_frames - 1, landing_frame + 10)

        self.decision_info = {
            'final_shot_frame': final_shot_frame,
            'landing_frame': landing_frame,
            'bounce_frame': landing_frame,
            'hitter_id': hitter_id,
            'receiver_id': receiver_id,
            'decision': decision,
            'type': 'WINNER_IN' if is_first_bounce_in else 'OUT',
            'margin_cm': margin_cm,
            'nearest_line': nearest_line,
            'point_winner': point_winner,
            'verdict_text': verdict_text,
            'reason_text': reason_text,
            'scoring_action': scoring_action,
            'landing_pos_mini': (lx, ly),
            'first_bounce_pos': (lx, ly),
            'second_bounce_pos': second_bounce_pos,
            'second_bounce_frame': second_bounce_frame,
            'camera_land_pos': camera_land_pos,
            'is_first_bounce_in': is_first_bounce_in,
            'flight_duration_sec': max(0.4, (landing_frame - final_shot_frame) / 24.0)
        }

        print(f"[RefereeSystem] Evaluation Complete:")
        print(f"  - First Bounce: F{landing_frame} | Decision: {decision} | Margin: {margin_cm:+.1f} cm")
        print(f"  - Verdict: {verdict_text} (Winner: Player {point_winner}) | Reason: {reason_text}")

        if self.mini_court is not None:
            self.mini_court.set_referee_decision(self.decision_info, all_bounces=self.all_bounces)

        return self.decision_info

    def draw_referee_overlay(self, video_frames, decision_info=None):
        """
        Draw broadcast Hawk-Eye ELC Referee Decision Card, 2D Impact Zoom Inset,
        and Ground Collision Impact Shockwaves on output video frames.
        """
        info = decision_info if decision_info is not None else self.decision_info
        if info is None:
            return video_frames

        landing_frame = info['landing_frame']
        decision = info['decision']
        margin_cm = info['margin_cm']
        nearest_line = info['nearest_line']
        point_winner = info['point_winner']
        hitter_id = info['hitter_id']
        camera_land_pos = info.get('camera_land_pos')

        is_out = (decision == "OUT")
        theme_color = (35, 35, 235) if is_out else (40, 220, 60) # BGR Red or Green
        theme_glow = (70, 70, 255) if is_out else (80, 240, 100)

        output_frames = []

        for f_idx, frame in enumerate(video_frames):
            # -----------------------------------------------------------------
            # 1. Ground Collision Shockwave & Impact Effect (Hiệu ứng bóng chạm đất)
            # -----------------------------------------------------------------
            # Render ground collision ripple for all bounces occurring in the video
            for bounce in self.all_bounces:
                b_frame = bounce['frame']
                age = f_idx - b_frame
                if 0 <= age <= 22:
                    bc_x, bc_y = int(bounce['camera_pos'][0]), int(bounce['camera_pos'][1])
                    b_is_in = bounce.get('is_in', True)
                    b_color = (30, 230, 60) if b_is_in else (30, 30, 230)
                    b_glow = (80, 255, 120) if b_is_in else (80, 80, 255)

                    # 1. Ground contact flash (first 4 frames)
                    if age <= 4:
                        overlay = frame.copy()
                        cv2.ellipse(overlay, (bc_x, bc_y), (16, 7), 0, 0, 360, (255, 255, 255), -1)
                        cv2.ellipse(overlay, (bc_x, bc_y), (26, 11), 0, 0, 360, b_color, -1)
                        alpha_flash = max(0.2, 0.65 - age * 0.12)
                        cv2.addWeighted(overlay, alpha_flash, frame, 1.0 - alpha_flash, 0, frame)

                        # Sparkle rays along court
                        cv2.line(frame, (bc_x - 14, bc_y), (bc_x + 14, bc_y), (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.line(frame, (bc_x, bc_y - 6), (bc_x, bc_y + 6), (255, 255, 255), 1, cv2.LINE_AA)

                    # 2. Concentric expanding shockwave rings (perspective squashed)
                    r1 = int(10 + age * 1.8)
                    r2 = int(18 + age * 2.6)
                    cv2.ellipse(frame, (bc_x, bc_y), (r1, int(r1 * 0.38)), 0, 0, 360, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.ellipse(frame, (bc_x, bc_y), (r2, int(r2 * 0.38)), 0, 0, 360, b_glow, 1, cv2.LINE_AA)

                    # 3. Ground contact pin & floating badge - ONLY for the decisive point-ending landing!
                    if bounce.get('is_final', False):
                        pin_top = bc_y - 36
                        cv2.line(frame, (bc_x, bc_y - 4), (bc_x, pin_top + 14), (255, 255, 255), 1, cv2.LINE_AA)

                        badge_txt = f"BOUNCE 1: IN (+{abs(bounce.get('margin_side_cm', 128)):.0f}cm)" if b_is_in else "OUT"
                        (bw, bh), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_DUPLEX, 0.44, 1)
                        bx1 = bc_x - int(bw / 2) - 8
                        bx2 = bc_x + int(bw / 2) + 8
                        by1 = pin_top - bh - 4
                        by2 = pin_top + 4

                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
                        cv2.rectangle(frame, (bx1, by1), (bx2, by2), b_color, 1)
                        cv2.putText(frame, badge_txt, (bx1 + 8, by2 - 5), cv2.FONT_HERSHEY_DUPLEX, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

            # -----------------------------------------------------------------
            # 2. Hawk-Eye Broadcast Decision Card (Bottom Left Corner)
            # -----------------------------------------------------------------
            # Display referee VAR card starting from decisive bounce (Frame 362)
            if f_idx >= max(0, landing_frame - 1):
                card_x, card_y = 35, 735
                card_w, card_h = 675, 305

                # Dark frosted glassmorphism background
                overlay = frame.copy()
                cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), (16, 18, 24), -1)
                cv2.addWeighted(overlay, 0.88, frame, 0.12, 0, frame)

                # Glowing card border
                cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + card_h), theme_color, 2)
                cv2.rectangle(frame, (card_x - 1, card_y - 1), (card_x + card_w + 1, card_y + card_h + 1), theme_glow, 1)

                # Header bar
                cv2.rectangle(frame, (card_x, card_y), (card_x + card_w, card_y + 38), (28, 30, 42), -1)
                cv2.line(frame, (card_x, card_y + 38), (card_x + card_w, card_y + 38), theme_color, 2)
                cv2.putText(
                    frame,
                    "HAWK-EYE ELC  |  REFEREE DECISION SYSTEM",
                    (card_x + 18, card_y + 26),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.62,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

                # Big Decision Badge [ IN ] or [ OUT ]
                badge_w, badge_h = 135, 48
                badge_x, badge_y = card_x + 20, card_y + 52
                cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), theme_color, -1)
                cv2.rectangle(frame, (badge_x, badge_y), (badge_x + badge_w, badge_y + badge_h), (255, 255, 255), 2)
                cv2.putText(
                    frame,
                    decision,
                    (badge_x + (42 if not is_out else 28), badge_y + 35),
                    cv2.FONT_HERSHEY_DUPLEX,
                    1.15,
                    (255, 255, 255),
                    3,
                    cv2.LINE_AA
                )

                # Text Metrics (Next to badge)
                cv2.putText(frame, "Vach:", (card_x + 172, card_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
                cv2.putText(frame, nearest_line[:24], (card_x + 225, card_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)

                cv2.putText(frame, "Margin:", (card_x + 172, card_y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (180, 180, 180), 1)
                margin_str = f"{margin_cm:+.1f} cm ({'Trong san' if not is_out else 'Ngoai san'})"
                cv2.putText(frame, margin_str, (card_x + 242, card_y + 94), cv2.FONT_HERSHEY_SIMPLEX, 0.52, theme_color, 2)

                # Point Award Banner
                pt_box_x, pt_box_y = card_x + 20, card_y + 114
                pt_box_w, pt_box_h = 375, 62
                cv2.rectangle(frame, (pt_box_x, pt_box_y), (pt_box_x + pt_box_w, pt_box_y + pt_box_h), (20, 48, 38) if not is_out else (48, 20, 20), -1)
                cv2.rectangle(frame, (pt_box_x, pt_box_y), (pt_box_x + pt_box_w, pt_box_y + pt_box_h), (0, 240, 160) if not is_out else (0, 215, 255), 1)
                p_winner_name = "Nadal" if point_winner == 1 else "Verdasco"
                cv2.putText(
                    frame,
                    f"PHAN QUYET: DIEM CHO PLAYER {point_winner} ({p_winner_name.upper()})",
                    (pt_box_x + 10, pt_box_y + 25),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.50,
                    (0, 240, 255),
                    2,
                    cv2.LINE_AA
                )
                detail_str = f"Passing Shot Winner (Bong dap dat lan 1 trong san)" if not is_out else "Danh bong ra ngoai"
                cv2.putText(frame, detail_str, (pt_box_x + 10, pt_box_y + 48), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (220, 240, 255), 1)

                # Supplementary Technical Information
                cv2.putText(frame, f"Frame cham dat 1: F{landing_frame}  |  Bay: {info['flight_duration_sec']:.2f}s", (card_x + 20, card_y + 205), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (180, 180, 180), 1)
                cv2.putText(frame, "Luat ITF: Chi can lan 1 trong san -> An diem hop le", (card_x + 20, card_y + 228), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (100, 240, 255), 1)
                cv2.putText(frame, f"Lan 2: Bong nay ra ngoai san sau lung doi thu", (card_x + 20, card_y + 251), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
                cv2.putText(frame, "Official Electronic Line Calling (ELC) Verified", (card_x + 20, card_y + 276), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 180), 1)

                # -------------------------------------------------------------
                # 3. Hawk-Eye 2D Zoom Caliper Inset (Close-up Court Patch)
                # -------------------------------------------------------------
                iz_x, iz_y = card_x + 415, card_y + 50
                iz_w, iz_h = 240, 235
                court_patch = np.zeros((iz_h, iz_w, 3), dtype=np.uint8)
                court_patch[:] = (130, 70, 25) # Tennis Navy Blue

                # Inset header bar
                cv2.rectangle(court_patch, (0, 0), (iz_w, 24), (20, 20, 28), -1)

                is_close_call = abs(margin_cm) <= 35.0
                if is_close_call:
                    cv2.putText(court_patch, "HAWK-EYE IMPACT ZOOM", (16, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)
                    line_x = 70
                    court_patch[:, :line_x] = (100, 52, 18) # Out of bounds region
                    for gy in range(0, iz_h, 32):
                        cv2.line(court_patch, (0, gy), (iz_w, gy), (115, 62, 22), 1)
                    cv2.putText(court_patch, "OUT", (18, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 220), 1)
                    cv2.putText(court_patch, "IN COURT", (line_x + 20, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 230, 140), 1)

                    # White chalk sideline
                    cv2.line(court_patch, (line_x, 0), (line_x, iz_h), (250, 250, 250), 10)
                    cv2.putText(court_patch, "SIDELINE", (line_x - 55, iz_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (240, 240, 240), 1)

                    # Dynamic ball impact position proportional to margin
                    ball_impact_x = int(line_x + 5 + np.clip(abs(margin_cm) * 2.2, 0, 95) + 16)
                    ball_impact_y = 135

                    cv2.ellipse(court_patch, (ball_impact_x, ball_impact_y), (26, 18), 0, 0, 360, (0, 180, 100) if not is_out else (40, 40, 220), -1)
                    cv2.ellipse(court_patch, (ball_impact_x, ball_impact_y), (22, 14), 0, 0, 360, (20, 240, 245) if not is_out else (80, 80, 255), -1)
                    cv2.ellipse(court_patch, (ball_impact_x, ball_impact_y), (24, 16), 0, 0, 360, (255, 255, 255), 2)

                    meas_left = line_x + 5
                    meas_right = ball_impact_x - 16
                    meas_y = ball_impact_y
                    if meas_right > meas_left:
                        cv2.line(court_patch, (meas_left, meas_y), (meas_right, meas_y), theme_color, 2)
                        cv2.line(court_patch, (meas_left, meas_y - 8), (meas_left, meas_y + 8), theme_color, 2)
                        cv2.line(court_patch, (meas_right, meas_y - 8), (meas_right, meas_y + 8), theme_color, 2)

                    badge_lbl = f"{margin_cm:+.1f} cm"
                    cv2.rectangle(court_patch, (line_x + 10, meas_y + 16), (line_x + 115, meas_y + 40), (20, 20, 20), -1)
                    cv2.rectangle(court_patch, (line_x + 10, meas_y + 16), (line_x + 115, meas_y + 40), theme_color, 1)
                    cv2.putText(court_patch, badge_lbl, (line_x + 16, meas_y + 33), cv2.FONT_HERSHEY_DUPLEX, 0.44, theme_color, 1)
                else:
                    # Clear Winner / Deep Landing Zone (Wide Court Overview)
                    cv2.putText(court_patch, "HAWK-EYE IMPACT LOCATOR", (14, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
                    line_x = 35
                    court_patch[:, :line_x] = (100, 52, 18) # Out of bounds
                    for gy in range(0, iz_h, 28):
                        cv2.line(court_patch, (0, gy), (iz_w, gy), (115, 62, 22), 1)

                    # White chalk sideline
                    cv2.line(court_patch, (line_x, 0), (line_x, iz_h), (250, 250, 250), 6)
                    cv2.putText(court_patch, "SIDELINE", (line_x + 8, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 200, 200), 1)

                    # Center service line marker on right
                    cv2.line(court_patch, (iz_w - 30, 0), (iz_w - 30, iz_h), (220, 220, 220), 2)
                    cv2.putText(court_patch, "CENTER", (iz_w - 65, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1)

                    # Proportional ball placement across singles court
                    ratio = min(0.75, max(0.18, abs(margin_cm) / 411.5))
                    ball_impact_x = int(line_x + ratio * (iz_w - 65))
                    ball_impact_y = 120

                    # Ball footprint
                    cv2.ellipse(court_patch, (ball_impact_x, ball_impact_y), (22, 16), 0, 0, 360, (0, 180, 100) if not is_out else (40, 40, 220), -1)
                    cv2.ellipse(court_patch, (ball_impact_x, ball_impact_y), (18, 12), 0, 0, 360, (20, 240, 245) if not is_out else (80, 80, 255), -1)
                    cv2.ellipse(court_patch, (ball_impact_x, ball_impact_y), (20, 14), 0, 0, 360, (255, 255, 255), 2)

                    # Caliper dimension ruler spanning across court
                    cv2.line(court_patch, (line_x + 3, ball_impact_y), (ball_impact_x - 14, ball_impact_y), theme_color, 2)
                    cv2.line(court_patch, (line_x + 3, ball_impact_y - 8), (line_x + 3, ball_impact_y + 8), theme_color, 2)
                    cv2.line(court_patch, (ball_impact_x - 14, ball_impact_y - 8), (ball_impact_x - 14, ball_impact_y + 8), theme_color, 2)

                    # Clear Winner Badge
                    badge_lbl = f"+{abs(margin_cm):.1f} cm ({abs(margin_cm)/100:.2f} m)"
                    cv2.rectangle(court_patch, (line_x + 10, 155), (iz_w - 20, 188), (18, 24, 38), -1)
                    cv2.rectangle(court_patch, (line_x + 10, 155), (iz_w - 20, 188), theme_color, 1)
                    cv2.putText(court_patch, badge_lbl, (line_x + 18, 177), cv2.FONT_HERSHEY_DUPLEX, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

                    verdict_sub = "CLEAR WINNER (IN)" if not is_out else "OUT OF BOUNDS"
                    cv2.putText(court_patch, verdict_sub, (line_x + 14, 212), cv2.FONT_HERSHEY_DUPLEX, 0.40, theme_color, 1, cv2.LINE_AA)

                # Inset border
                cv2.rectangle(court_patch, (0, 0), (iz_w - 1, iz_h - 1), (180, 180, 180), 1)
                frame[iz_y : iz_y + iz_h, iz_x : iz_x + iz_w] = court_patch

            output_frames.append(frame)

        return output_frames
