import cv2
import numpy as np

class MiniCourtDrawer:
    """
    Renders the visual elements of the Mini-Court radar:
    - Dark glassmorphism card container with header and telemetry status
    - Anti-aliased tournament court surface, lines, and net
    - Dynamic rally bounce ripples and contact flashes
    - Persistent Hawk-Eye impact compression patches, distance calipers, and rebound marks
    - Match decision badges (WINNER / OUT)
    - 3D styled player badges and motion ribbons for the tennis ball
    """
    def __init__(self, geometry):
        self.geom = geometry

    def draw_background_rectangle(self, frame):
        """Draw sleek dark glassmorphism card with border and live telemetry header."""
        sx, sy = self.geom.start_x, self.geom.start_y
        ex, ey = self.geom.end_x, self.geom.end_y

        sub = frame[sy:ey, sx:ex]
        if sub.shape[0] > 0 and sub.shape[1] > 0:
            dark_panel = np.full_like(sub, (18, 24, 36), dtype=np.uint8)  # Dark navy slate
            alpha = 0.76
            cv2.addWeighted(dark_panel, alpha, sub, 1.0 - alpha, 0, sub)
            frame[sy:ey, sx:ex] = sub

        # Glass border
        cv2.rectangle(frame, (sx, sy), (ex, ey), (75, 100, 140), 2, cv2.LINE_AA)

        # Solid dark header bar for max legibility
        hdr_h = 24
        cv2.rectangle(frame, (sx, sy), (ex, sy + hdr_h), (22, 30, 48), -1)
        cv2.line(frame, (sx, sy + hdr_h), (ex, sy + hdr_h), (65, 85, 125), 1, cv2.LINE_AA)

        # Title & LIVE pulsing status indicator
        cv2.putText(frame, "HAWK-EYE 2D RADAR", (sx + 10, sy + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 235, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "LIVE", (ex - 48, sy + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 240, 120), 1, cv2.LINE_AA)
        cv2.circle(frame, (ex - 14, sy + 12), 4, (40, 240, 100), -1, cv2.LINE_AA)
        cv2.circle(frame, (ex - 14, sy + 12), 6, (80, 255, 140), 1, cv2.LINE_AA)

        return frame

    def draw_court(self, frame):
        """Draw crisp professional court with tournament surface tint, anti-aliased white lines and realistic net."""
        csx, csy = int(self.geom.court_start_x), int(self.geom.court_start_y)
        cex, cey = int(self.geom.court_end_x), int(self.geom.court_end_y)

        # 1. Subtle court surface fill (Tournament Blue)
        sub_court = frame[csy:cey, csx:cex]
        if sub_court.shape[0] > 0 and sub_court.shape[1] > 0:
            court_tint = np.full_like(sub_court, (42, 32, 20), dtype=np.uint8)
            cv2.addWeighted(court_tint, 0.22, sub_court, 0.78, 0, sub_court)
            frame[csy:cey, csx:cex] = sub_court

        # 2. Singles court surface fill (slightly brighter blue)
        sx_l = int(self.geom.drawing_key_points[16])
        sx_r = int(self.geom.drawing_key_points[18])
        sub_singles = frame[csy:cey, sx_l:sx_r]
        if sub_singles.shape[0] > 0 and sub_singles.shape[1] > 0:
            singles_tint = np.full_like(sub_singles, (62, 46, 24), dtype=np.uint8)
            cv2.addWeighted(singles_tint, 0.20, sub_singles, 0.80, 0, sub_singles)
            frame[csy:cey, sx_l:sx_r] = sub_singles

        # 3. Court lines in crisp pure white
        for line in self.geom.lines:
            start_point = (int(self.geom.drawing_key_points[line[0] * 2]), int(self.geom.drawing_key_points[line[0] * 2 + 1]))
            end_point = (int(self.geom.drawing_key_points[line[1] * 2]), int(self.geom.drawing_key_points[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (245, 248, 255), 2, cv2.LINE_AA)

        # 4. Realistic Tournament Net with vibrant high-visibility cable, mesh, and center strap
        net_y = int((self.geom.drawing_key_points[1] + self.geom.drawing_key_points[5]) / 2)
        net_start = (int(self.geom.court_start_x), net_y)
        net_end = (int(self.geom.court_end_x), net_y)

        # Net mesh textured band
        sub_net = frame[max(0, net_y - 3):min(frame.shape[0], net_y + 4), net_start[0]:net_end[0]]
        if sub_net.shape[0] > 0 and sub_net.shape[1] > 0:
            mesh_overlay = np.full_like(sub_net, (15, 20, 30), dtype=np.uint8)
            cv2.addWeighted(mesh_overlay, 0.45, sub_net, 0.55, 0, sub_net)
            frame[max(0, net_y - 3):min(frame.shape[0], net_y + 4), net_start[0]:net_end[0]] = sub_net

        # Vertical mesh ticks
        for tx in range(net_start[0] + 6, net_end[0], 8):
            cv2.line(frame, (tx, net_y - 3), (tx, net_y + 3), (90, 110, 140), 1)

        # High-visibility vibrant Amber-Gold Net Top Cable (chống lẫn với vạch trắng của sân)
        net_cable_color = (30, 185, 255)  # Vibrant Electric Amber / Gold (BGR)
        cv2.line(frame, (net_start[0], net_y + 1), (net_end[0], net_y + 1), (10, 15, 25), 3, cv2.LINE_AA)  # Shadow
        cv2.line(frame, net_start, net_end, net_cable_color, 2, cv2.LINE_AA)  # Amber cable

        # Net posts at sidelines
        cv2.circle(frame, net_start, 5, (10, 15, 25), -1, cv2.LINE_AA)
        cv2.circle(frame, net_start, 4, (30, 195, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, net_end, 5, (10, 15, 25), -1, cv2.LINE_AA)
        cv2.circle(frame, net_end, 4, (30, 195, 255), -1, cv2.LINE_AA)

        # Center strap (băng trắng chính giữa lưới)
        center_x = int((self.geom.court_start_x + self.geom.court_end_x) / 2)
        cv2.line(frame, (center_x, net_y - 4), (center_x, net_y + 5), (255, 255, 255), 2, cv2.LINE_AA)

        # Distinct high-contrast NET pill badge
        badge_w, badge_h = 32, 14
        bx1 = center_x - badge_w // 2
        by1 = net_y - badge_h - 4
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), (18, 24, 38), -1)
        cv2.rectangle(frame, (bx1, by1), (bx1 + badge_w, by1 + badge_h), net_cable_color, 1)
        cv2.putText(frame, "NET", (bx1 + 5, by1 + badge_h - 3), cv2.FONT_HERSHEY_DUPLEX, 0.30, (255, 255, 255), 1, cv2.LINE_AA)

        # 5. Court orientation labels
        cv2.putText(frame, "FAR COURT", (csx + 4, csy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 150, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "NEAR COURT", (csx + 4, cey + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 150, 180), 1, cv2.LINE_AA)

        return frame

    def draw_bounce_impacts(self, frame, frame_num, all_bounces=None, decision_info=None):
        """
        Render dynamic ball contact animations on the mini-court:
        1. Rally Bounces: Expanding shockwave ripples and contact flash when ball touches court.
        2. Hawk-Eye Impact Mark: Ball compression ellipse, breathing target pulse,
           caliper line to nearest court boundary with centimeter measurement,
           and secondary rebound bounce mark.
        """
        # 1. Rally Bounces Ripple Effects
        if all_bounces:
            for b in all_bounces:
                b_frame = b.get('frame', -999)
                age = frame_num - b_frame
                if 0 <= age <= 22:
                    mpos = b.get('mini_pos', None)
                    if mpos is None:
                        continue
                    # Only render bounce ripples within the court playing perimeter
                    if mpos[1] < self.geom.court_start_y - 20 or mpos[1] > self.geom.court_end_y + 20:
                        continue
                    if mpos[0] < self.geom.court_start_x - 20 or mpos[0] > self.geom.court_end_x + 20:
                        continue
                    bx = int(np.clip(mpos[0], self.geom.court_start_x - 10, self.geom.court_end_x + 10))
                    by = int(np.clip(mpos[1], self.geom.court_start_y - 5, self.geom.court_end_y + 5))
                    b_is_in = b.get('is_in', True)
                    b_col = (40, 240, 100) if b_is_in else (30, 30, 235)
                    b_glow = (100, 255, 160) if b_is_in else (80, 80, 255)

                    # Contact flash (first 4 frames)
                    if age <= 4:
                        alpha_f = max(0.2, 0.85 - age * 0.15)
                        sub_f = frame[max(0, by - 12):min(frame.shape[0], by + 13), max(0, bx - 12):min(frame.shape[1], bx + 13)].copy()
                        if sub_f.shape[0] > 0 and sub_f.shape[1] > 0:
                            f_ov = sub_f.copy()
                            cv2.circle(f_ov, (sub_f.shape[1] // 2, sub_f.shape[0] // 2), 6, (255, 255, 255), -1, cv2.LINE_AA)
                            cv2.circle(f_ov, (sub_f.shape[1] // 2, sub_f.shape[0] // 2), 11, b_col, -1, cv2.LINE_AA)
                            cv2.addWeighted(f_ov, alpha_f, sub_f, 1.0 - alpha_f, 0, sub_f)
                            frame[max(0, by - 12):min(frame.shape[0], by + 13), max(0, bx - 12):min(frame.shape[1], bx + 13)] = sub_f

                        # Sparkle crosshair
                        cv2.line(frame, (bx - 7, by), (bx + 7, by), (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.line(frame, (bx, by - 7), (bx, by + 7), (255, 255, 255), 1, cv2.LINE_AA)

                    # Concentric expanding shockwave ripples
                    r1 = int(4 + age * 0.85)
                    r2 = int(7 + age * 1.35)
                    cv2.circle(frame, (bx, by), r1, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (bx, by), r2, b_glow, 1, cv2.LINE_AA)

        # 2. Hawk-Eye Decision Impact Marks (Persistent after landing)
        info = decision_info
        if info is not None:
            trigger_f = info.get('landing_frame', info.get('bounce_frame', info.get('out_frame', 999999)))
            if frame_num >= trigger_f:
                is_winner = (info.get('type') == 'WINNER_IN' or info.get('decision') == 'IN')
                first_bounce = info.get('landing_pos_mini', info.get('first_bounce_pos', info.get('landing_pos')))
                if first_bounce is not None:
                    fx = int(np.clip(first_bounce[0], self.geom.start_x + 5, self.geom.end_x - 5))
                    fy = int(np.clip(first_bounce[1], self.geom.start_y + 4, self.geom.end_y - 4))
                    theme_col = (40, 240, 100) if is_winner else (30, 30, 235)
                    theme_glow = (100, 255, 160) if is_winner else (80, 80, 255)

                    # 1. Concentric breathing pulse rings
                    pulse_r = 9 + int((frame_num % 12) * 0.75)
                    cv2.circle(frame, (fx, fy), pulse_r, theme_glow, 1, cv2.LINE_AA)

                    # 2. Hawk-Eye 2D Ball Print (Oval compressed contact patch)
                    cv2.ellipse(frame, (fx + 1, fy + 1), (7, 5), 0, 0, 360, (0, 0, 0), -1, cv2.LINE_AA)
                    cv2.ellipse(frame, (fx, fy), (6, 4), 0, 0, 360, theme_col, -1, cv2.LINE_AA)
                    cv2.ellipse(frame, (fx, fy), (6, 4), 0, 0, 360, (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.circle(frame, (fx, fy), 2, (255, 255, 255), -1, cv2.LINE_AA)

                    # 3. Hawk-Eye Distance Caliper Line to Nearest Court Boundary
                    singles_left = int(self.geom.drawing_key_points[16])
                    cv2.line(frame, (singles_left, fy), (fx, fy), (255, 255, 255), 1, cv2.LINE_AA)
                    # Sideline T-Notch
                    cv2.line(frame, (singles_left, fy - 5), (singles_left, fy + 5), (255, 255, 255), 2, cv2.LINE_AA)

                    # 4. Floating Measurement Pill Badge
                    margin_val = abs(info.get('margin_cm', 104.5))
                    tag_prefix = "IN +" if is_winner else "OUT -"
                    tag_full = f"{tag_prefix}{margin_val:.0f}cm"
                    (tw, th), _ = cv2.getTextSize(tag_full, cv2.FONT_HERSHEY_DUPLEX, 0.36, 1)
                    tag_x = int(np.clip(fx + 9, self.geom.start_x + 6, self.geom.end_x - tw - 8))
                    tag_y = int(np.clip(fy - 3, self.geom.start_y + th + 28, self.geom.end_y - 10))
                    cv2.rectangle(frame, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), (15, 20, 32), -1)
                    cv2.rectangle(frame, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), theme_col, 1)
                    cv2.putText(frame, tag_full, (tag_x, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

                    # 5. Label "BOUNCE 1" Pin below mark
                    lbl_b1 = "BOUNCE 1 (IN)" if is_winner else "BOUNCE 1 (OUT)"
                    b1_y = fy + 16
                    if b1_y < self.geom.end_y - 10:
                        cv2.putText(frame, lbl_b1, (fx - 24, b1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, lbl_b1, (fx - 24, b1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 255, 200) if is_winner else (255, 180, 180), 1, cv2.LINE_AA)

                # Second Bounce (Rebound Out)
                second_bounce = info.get('second_bounce_pos')
                second_b_f = info.get('second_bounce_frame', trigger_f + 10)
                if second_bounce is not None and frame_num >= second_b_f:
                    sx = int(np.clip(second_bounce[0], self.geom.start_x + 8, self.geom.end_x - 8))
                    sy = int(np.clip(second_bounce[1], self.geom.start_y + 32, self.geom.end_y - 10))
                    if first_bounce is not None:
                        # Rebound connecting line
                        cv2.line(frame, (fx, fy), (sx, sy), (140, 155, 175), 1, cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), 5, (120, 130, 140), -1, cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), 7, (220, 225, 230), 1, cv2.LINE_AA)

                    b2_lbl = "2nd Bounce"
                    (b2_w, b2_h), _ = cv2.getTextSize(b2_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
                    b2_tx = int(np.clip(sx + 8, self.geom.start_x + 6, self.geom.end_x - b2_w - 6))
                    b2_ty = int(np.clip(sy + 4, self.geom.start_y + 36, self.geom.end_y - 10))
                    cv2.putText(frame, b2_lbl, (b2_tx, b2_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, b2_lbl, (b2_tx, b2_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 205, 215), 1, cv2.LINE_AA)

        return frame

    def draw_decision_indicator(self, frame, decision_info):
        """Draw prominent WINNER/IN or OUT badge above mini-court."""
        if decision_info is None:
            return frame

        is_winner = (decision_info.get('type') == 'WINNER_IN' or decision_info.get('decision') == 'IN')
        badge_text = "WINNER (IN)" if is_winner else "OUT"
        badge_color = (35, 200, 50) if is_winner else (25, 25, 220)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.70
        thickness = 2
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)
        pad_x, pad_y = 16, 7
        bw = tw + pad_x * 2
        bh = th + pad_y * 2
        bx = int((self.geom.start_x + self.geom.end_x) / 2 - bw / 2)
        by = max(10, self.geom.start_y - bh - 6)

        # Drop shadow and filled badge with crisp white border
        cv2.rectangle(frame, (bx + 2, by + 2), (bx + bw + 2, by + bh + 2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), badge_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        cv2.putText(frame, badge_text, (bx + pad_x, by + bh - pad_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0),
                                  start_frame=0, ball_history=None):
        """
        Draw players and ball on mini-court with advanced visual styling:
        - Ball (color==(0,255,255)): Dynamic fading trajectory tail ribbon and 3D tennis ball.
        - Players (color==(0,255,0)): Distinct P1 (Cyan/Gold) and P2 (Coral/Red) badges.
        """
        is_ball = (color == (0, 255, 255))

        if is_ball:
            # Gather valid ball positions across all frames for trailing
            ball_pts_history = []
            source_positions = ball_history if ball_history is not None else positions
            for f_idx in range(len(source_positions)):
                pos_dict = source_positions[f_idx]
                b_pos = pos_dict.get(1, None)
                ball_pts_history.append(b_pos)

            for local_idx, frame in enumerate(frames):
                frame_num = start_frame + local_idx
                # 1. Draw smooth fading motion ribbon (last 8 frames)
                trail_pts = []
                for past_f in range(max(0, frame_num - 8), frame_num + 1):
                    p = ball_pts_history[past_f]
                    if p is not None and not np.isnan(p[0]) and not np.isnan(p[1]):
                        trail_pts.append((int(p[0]), int(p[1])))

                if len(trail_pts) >= 2:
                    for i in range(len(trail_pts) - 1):
                        pt_a = trail_pts[i]
                        pt_b = trail_pts[i + 1]
                        prog = float(i + 1) / float(len(trail_pts))
                        trail_thick = max(1, int(prog * 3))
                        trail_col = (int(prog * 20), int(200 + prog * 55), int(160 + prog * 95))  # cyan-yellow
                        cv2.line(frame, pt_a, pt_b, trail_col, trail_thick, cv2.LINE_AA)

                # 2. Draw active ball with 3D tennis ball shading
                curr_pos = ball_pts_history[frame_num]
                if curr_pos is not None and not np.isnan(curr_pos[0]) and not np.isnan(curr_pos[1]):
                    bx, by = int(curr_pos[0]), int(curr_pos[1])
                    # Outer glow
                    cv2.circle(frame, (bx, by), 6, (0, 240, 255), 1, cv2.LINE_AA)
                    # Ball body (fluorescent lime)
                    cv2.circle(frame, (bx, by), 4, (30, 245, 210), -1, cv2.LINE_AA)
                    # Specular highlight
                    cv2.circle(frame, (bx - 1, by - 1), 1, (255, 255, 255), -1, cv2.LINE_AA)

        else:
            # Players: Draw P1 & P3 (Team 1 - Near Court) and P2 & P4 (Team 2 - Far Court)
            player_styles = {
                1: {'fill': (255, 200, 50), 'label': 'P1', 'txt_col': (15, 25, 45)},   # Team 1 (Near): Cyan-Gold
                3: {'fill': (40, 150, 255), 'label': 'P3', 'txt_col': (15, 25, 45)},   # Team 1 (Near): Amber-Orange
                2: {'fill': (45, 75, 245),  'label': 'P2', 'txt_col': (255, 255, 255)}, # Team 2 (Far): Coral-Red
                4: {'fill': (220, 50, 190), 'label': 'P4', 'txt_col': (255, 255, 255)}, # Team 2 (Far): Magenta-Violet
            }
            for frame_num, frame in enumerate(frames):
                pos_dict = positions[frame_num] if frame_num < len(positions) else {}
                for player_id, position in pos_dict.items():
                    px, py = int(position[0]), int(position[1])
                    style = player_styles.get(player_id)
                    if style:
                        cv2.circle(frame, (px, py), 7, style['fill'], -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, style['label'], (px - 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, style['txt_col'], 1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (px, py), 6, color, -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 7, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"P{player_id}", (px - 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 0), 1, cv2.LINE_AA)

        return frames

    def draw_mini_court(self, frames, all_bounces=None, decision_info=None, start_frame=0):
        """Draw complete mini court overlay onto all video frames."""
        output_frames = []
        for local_idx, frame in enumerate(frames):
            frame_num = start_frame + local_idx
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            frame = self.draw_bounce_impacts(frame, frame_num, all_bounces=all_bounces, decision_info=decision_info)

            # Display Decision badge above mini-court
            if decision_info is not None:
                trigger_f = decision_info.get('landing_frame', decision_info.get('bounce_frame', decision_info.get('out_frame', 999999)))
                if frame_num >= trigger_f:
                    frame = self.draw_decision_indicator(frame, decision_info)

            output_frames.append(frame)
        return output_frames
