import cv2
import numpy as np
import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    get_foot_position,
    get_closest_keypoint_index,
    get_height_of_bbox,
    measure_xy_distance,
    get_center_of_bbox,
    measure_distance
)

class MiniCourt():
    def __init__(self,frame):
        self.drawing_rectangle_width = 250
        self.drawing_rectangle_height = 500
        self.buffer = 50
        self.padding_court=20

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()
        self.ball_out_info = None
        self.ball_decision_info = None
        self.all_bounces = []


    def set_referee_decision(self, decision_info, all_bounces=None):
        """Register the referee decision info and all detected rally bounces for mini-court visualization."""
        self.ball_decision_info = decision_info
        if all_bounces is not None:
            self.all_bounces = all_bounces

    def convert_meters_to_pixels(self, meters):
        return convert_meters_to_pixel_distance(meters,
                                                constants.DOUBLE_LINE_WIDTH,
                                                self.court_drawing_width
                                            )

    def set_court_drawing_key_points(self):
        drawing_key_points = [0]*28

        # point 0 
        drawing_key_points[0] , drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1
        drawing_key_points[2] , drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT*2)
        # point 3
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5] 
        # #point 4
        drawing_key_points[8] = drawing_key_points[0] +  self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1] 
        # #point 5
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5] 
        # #point 6
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3] 
        # #point 7
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7] 
        # #point 8
        drawing_key_points[16] = drawing_key_points[8] 
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 9
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17] 
        # #point 10
        drawing_key_points[20] = drawing_key_points[10] 
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # # #point 11
        drawing_key_points[22] = drawing_key_points[20] +  self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21] 
        # # #point 12
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18])/2)
        drawing_key_points[25] = drawing_key_points[17] 
        # # #point 13
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22])/2)
        drawing_key_points[27] = drawing_key_points[21] 

        self.drawing_key_points=drawing_key_points

    def set_court_lines(self):
        self.lines = [
            (0, 2),
            (4, 5),
            (6,7),
            (1,3),
            
            (0,1),
            (8,9),
            (10,11),
            (10,11),
            (2,3)
        ]

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self,frame):
        frame= frame.copy()

        self.end_x = frame.shape[1] - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def draw_background_rectangle(self, frame):
        """Draw sleek dark glassmorphism card with border and live telemetry header."""
        sub = frame[self.start_y:self.end_y, self.start_x:self.end_x]
        if sub.shape[0] > 0 and sub.shape[1] > 0:
            dark_panel = np.full_like(sub, (18, 24, 36), dtype=np.uint8) # Dark navy slate
            alpha = 0.76
            cv2.addWeighted(dark_panel, alpha, sub, 1.0 - alpha, 0, sub)
            frame[self.start_y:self.end_y, self.start_x:self.end_x] = sub

        # Glass border
        cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.end_y), (75, 100, 140), 2, cv2.LINE_AA)

        # Solid dark header bar for max legibility
        hdr_h = 24
        cv2.rectangle(frame, (self.start_x, self.start_y), (self.end_x, self.start_y + hdr_h), (22, 30, 48), -1)
        cv2.line(frame, (self.start_x, self.start_y + hdr_h), (self.end_x, self.start_y + hdr_h), (65, 85, 125), 1, cv2.LINE_AA)

        # Title & LIVE pulsing status indicator
        cv2.putText(frame, "HAWK-EYE 2D RADAR", (self.start_x + 10, self.start_y + 16), cv2.FONT_HERSHEY_DUPLEX, 0.38, (220, 235, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, "LIVE", (self.end_x - 48, self.start_y + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (80, 240, 120), 1, cv2.LINE_AA)
        cv2.circle(frame, (self.end_x - 14, self.start_y + 12), 4, (40, 240, 100), -1, cv2.LINE_AA)
        cv2.circle(frame, (self.end_x - 14, self.start_y + 12), 6, (80, 255, 140), 1, cv2.LINE_AA)

        return frame

    def draw_court(self, frame):
        """Draw crisp professional court with tournament surface tint, anti-aliased white lines and realistic net."""
        # 1. Subtle court surface fill (Tournament Blue)
        csx, csy = int(self.court_start_x), int(self.court_start_y)
        cex, cey = int(self.court_end_x), int(self.court_end_y)
        sub_court = frame[csy:cey, csx:cex]
        if sub_court.shape[0] > 0 and sub_court.shape[1] > 0:
            court_tint = np.full_like(sub_court, (42, 32, 20), dtype=np.uint8) # Dark blue
            cv2.addWeighted(court_tint, 0.22, sub_court, 0.78, 0, sub_court)
            frame[csy:cey, csx:cex] = sub_court

        # 2. Singles court surface fill (slightly brighter blue)
        sx_l = int(self.drawing_key_points[16])
        sx_r = int(self.drawing_key_points[18])
        sub_singles = frame[csy:cey, sx_l:sx_r]
        if sub_singles.shape[0] > 0 and sub_singles.shape[1] > 0:
            singles_tint = np.full_like(sub_singles, (62, 46, 24), dtype=np.uint8)
            cv2.addWeighted(singles_tint, 0.20, sub_singles, 0.80, 0, sub_singles)
            frame[csy:cey, sx_l:sx_r] = sub_singles

        # 3. Court lines in crisp pure white
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (245, 248, 255), 2, cv2.LINE_AA)

        # 4. Realistic Net with center strap and posts
        net_y = int((self.drawing_key_points[1] + self.drawing_key_points[5]) / 2)
        net_start = (int(self.court_start_x), net_y)
        net_end = (int(self.court_end_x), net_y)
        cv2.line(frame, (net_start[0], net_y + 1), (net_end[0], net_y + 1), (20, 20, 25), 2, cv2.LINE_AA) # shadow
        cv2.line(frame, net_start, net_end, (220, 225, 235), 2, cv2.LINE_AA) # white net band
        cv2.circle(frame, net_start, 4, (240, 180, 50), -1, cv2.LINE_AA) # net posts
        cv2.circle(frame, net_end, 4, (240, 180, 50), -1, cv2.LINE_AA)
        cv2.putText(frame, "NET", (int((self.court_start_x + self.court_end_x)/2) - 10, net_y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (170, 185, 205), 1, cv2.LINE_AA)

        # 5. Court orientation labels
        cv2.putText(frame, "FAR COURT", (csx + 4, csy - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 150, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, "NEAR COURT", (csx + 4, cey + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (130, 150, 180), 1, cv2.LINE_AA)

        return frame

    def draw_bounce_impacts(self, frame, frame_num):
        """
        Render dynamic ball contact animations on the mini-court:
        1. Rally Bounces: Expanding shockwave ripples and contact flash when ball touches court.
        2. Hawk-Eye Impact Mark: Ball compression ellipse, breathing target pulse,
           caliper line to nearest court boundary with centimeter measurement,
           and secondary rebound bounce mark.
        """
        # 1. Rally Bounces Ripple Effects
        if self.all_bounces:
            for b in self.all_bounces:
                b_frame = b.get('frame', -999)
                age = frame_num - b_frame
                if 0 <= age <= 18:
                    mpos = b.get('mini_pos', None)
                    if mpos is None:
                        continue
                    # Only render bounce ripples within the court playing perimeter
                    if mpos[1] < self.court_start_y - 10 or mpos[1] > self.court_end_y + 10:
                        continue
                    if mpos[0] < self.court_start_x - 10 or mpos[0] > self.court_end_x + 10:
                        continue
                    bx = int(np.clip(mpos[0], self.court_start_x - 5, self.court_end_x + 5))
                    by = int(np.clip(mpos[1], self.court_start_y, self.court_end_y))
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
        info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
        if info is not None:
            trigger_f = info.get('landing_frame', info.get('bounce_frame', info.get('out_frame', 999999)))
            if frame_num >= trigger_f:
                is_winner = (info.get('type') == 'WINNER_IN' or info.get('decision') == 'IN')
                first_bounce = info.get('landing_pos_mini', info.get('first_bounce_pos', info.get('landing_pos')))
                if first_bounce is not None:
                    fx = int(np.clip(first_bounce[0], self.start_x + 5, self.end_x - 5))
                    fy = int(np.clip(first_bounce[1], self.start_y + 4, self.end_y - 4))
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
                    # Left singles sideline:
                    singles_left = int(self.drawing_key_points[16])
                    cv2.line(frame, (singles_left, fy), (fx, fy), (255, 255, 255), 1, cv2.LINE_AA)
                    # Sideline T-Notch
                    cv2.line(frame, (singles_left, fy - 5), (singles_left, fy + 5), (255, 255, 255), 2, cv2.LINE_AA)

                    # 4. Floating Measurement Pill Badge
                    margin_val = abs(info.get('margin_cm', 104.5))
                    tag_prefix = "IN +" if is_winner else "OUT -"
                    tag_full = f"{tag_prefix}{margin_val:.0f}cm"
                    (tw, th), _ = cv2.getTextSize(tag_full, cv2.FONT_HERSHEY_DUPLEX, 0.36, 1)
                    tag_x = int(np.clip(fx + 9, self.start_x + 6, self.end_x - tw - 8))
                    tag_y = int(np.clip(fy - 3, self.start_y + th + 28, self.end_y - 10))
                    cv2.rectangle(frame, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), (15, 20, 32), -1)
                    cv2.rectangle(frame, (tag_x - 3, tag_y - th - 3), (tag_x + tw + 3, tag_y + 3), theme_col, 1)
                    cv2.putText(frame, tag_full, (tag_x, tag_y), cv2.FONT_HERSHEY_DUPLEX, 0.36, (255, 255, 255), 1, cv2.LINE_AA)

                    # 5. Label "BOUNCE 1" Pin below mark
                    lbl_b1 = "BOUNCE 1 (IN)" if is_winner else "BOUNCE 1 (OUT)"
                    b1_y = fy + 16
                    if b1_y < self.end_y - 10:
                        cv2.putText(frame, lbl_b1, (fx - 24, b1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
                        cv2.putText(frame, lbl_b1, (fx - 24, b1_y), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (180, 255, 200) if is_winner else (255, 180, 180), 1, cv2.LINE_AA)

                # Second Bounce (Rebound Out)
                second_bounce = info.get('second_bounce_pos')
                second_b_f = info.get('second_bounce_frame', trigger_f + 10)
                if second_bounce is not None and frame_num >= second_b_f:
                    sx = int(np.clip(second_bounce[0], self.start_x + 8, self.end_x - 8))
                    sy = int(np.clip(second_bounce[1], self.start_y + 32, self.end_y - 10))
                    if first_bounce is not None:
                        # Rebound connecting line
                        cv2.line(frame, (fx, fy), (sx, sy), (140, 155, 175), 1, cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), 5, (120, 130, 140), -1, cv2.LINE_AA)
                    cv2.circle(frame, (sx, sy), 7, (220, 225, 230), 1, cv2.LINE_AA)

                    b2_lbl = "2nd Bounce"
                    (b2_w, b2_h), _ = cv2.getTextSize(b2_lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.30, 1)
                    b2_tx = int(np.clip(sx + 8, self.start_x + 6, self.end_x - b2_w - 6))
                    b2_ty = int(np.clip(sy + 4, self.start_y + 36, self.end_y - 10))
                    cv2.putText(frame, b2_lbl, (b2_tx, b2_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, b2_lbl, (b2_tx, b2_ty), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (200, 205, 215), 1, cv2.LINE_AA)

        return frame

    def draw_decision_indicator(self, frame):
        """Draw prominent WINNER/IN or OUT badge above mini-court."""
        info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
        if info is None:
            return frame

        is_winner = (info.get('type') == 'WINNER_IN' or info.get('decision') == 'IN')
        badge_text = "WINNER (IN)" if is_winner else "OUT"
        badge_color = (35, 200, 50) if is_winner else (25, 25, 220)

        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 0.70
        thickness = 2
        (tw, th), _ = cv2.getTextSize(badge_text, font, font_scale, thickness)
        pad_x, pad_y = 16, 7
        bw = tw + pad_x * 2
        bh = th + pad_y * 2
        bx = int((self.start_x + self.end_x) / 2 - bw / 2)
        by = max(10, self.start_y - bh - 6)

        # Drop shadow and filled badge with crisp white border
        cv2.rectangle(frame, (bx + 2, by + 2), (bx + bw + 2, by + bh + 2), (0, 0, 0), -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), badge_color, -1)
        cv2.rectangle(frame, (bx, by), (bx + bw, by + bh), (255, 255, 255), 2)
        cv2.putText(frame, badge_text, (bx + pad_x, by + bh - pad_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

        return frame

    def draw_mini_court(self, frames):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
            frame = self.draw_bounce_impacts(frame, frame_num)

            # Display Decision badge above mini-court
            active_info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
            if active_info is not None:
                trigger_f = active_info.get('landing_frame', active_info.get('bounce_frame', active_info.get('out_frame', 999999)))
                if frame_num >= trigger_f:
                    frame = self.draw_decision_indicator(frame)

            output_frames.append(frame)
        return output_frames

    def get_start_point_of_mini_court(self):
        return (self.court_start_x,self.court_start_y)
    def get_width_of_mini_court(self):
        return self.court_drawing_width
    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters
                                   ):
        
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(distance_from_keypoint_x_pixels,
                                                                           player_height_in_meters,
                                                                           player_height_in_pixels
                                                                           )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(distance_from_keypoint_y_pixels,
                                                                                player_height_in_meters,
                                                                                player_height_in_pixels
                                                                          )
        
        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_coourt_keypoint = ( self.drawing_key_points[closest_key_point_index*2],
                                        self.drawing_key_points[closest_key_point_index*2+1]
                                        )
        
        mini_court_player_position = (closest_mini_coourt_keypoint[0]+mini_court_x_distance_pixels,
                                      closest_mini_coourt_keypoint[1]+mini_court_y_distance_pixels
                                        )

        return  mini_court_player_position

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points, ball_shot_frames=None):
        """
        Convert player and ball bounding boxes to mini-court coordinates using
        2D Homography perspective transformation and Physics-Informed 3D Ground Projection.
        Prevents the perspective distortion where an airborne ball (high in camera view)
        gets mistakenly projected far beyond the opponent's baseline.
        """
        import pandas as pd
        dst_pts = np.array([(self.drawing_key_points[2 * i], self.drawing_key_points[2 * i + 1]) for i in range(14)], dtype=np.float32)

        is_dynamic = (
            isinstance(original_court_key_points, (list, np.ndarray))
            and len(original_court_key_points) == len(player_boxes)
            and hasattr(original_court_key_points[0], '__len__')
            and len(original_court_key_points[0]) >= 28
        )

        def compute_h(kps):
            src_pts = np.array([(kps[2 * i], kps[2 * i + 1]) for i in range(14)], dtype=np.float32)
            h_mat, _ = cv2.findHomography(src_pts, dst_pts)
            return h_mat

        if is_dynamic:
            h_matrices = [compute_h(kps) for kps in original_court_key_points]
            H_static = h_matrices[0]
        else:
            H_static = compute_h(original_court_key_points)
            h_matrices = None

        output_player_boxes = []
        output_ball_boxes = []

        min_x = self.start_x + 5
        max_x = self.end_x - 5
        min_y = self.start_y + 5
        max_y = self.end_y - 5

        # 1. Project Players (using foot position on court ground)
        for frame_num, player_dict in enumerate(player_boxes):
            H = h_matrices[frame_num] if is_dynamic else H_static
            output_player_bboxes_dict = {}
            for player_id, bbox in player_dict.items():
                foot_position = get_foot_position(bbox)
                pt_arr = np.array([[[foot_position[0], foot_position[1]]]], dtype=np.float32)
                proj = cv2.perspectiveTransform(pt_arr, H)[0][0]
                px = float(np.clip(proj[0], min_x, max_x))
                py = float(np.clip(proj[1], min_y, max_y))
                output_player_bboxes_dict[player_id] = (px, py)
            output_player_boxes.append(output_player_bboxes_dict)

        # Extract continuous player tracks
        num_frames = len(player_boxes)
        p1_pts = [output_player_boxes[f].get(1, (np.nan, np.nan)) for f in range(num_frames)]
        p2_pts = [output_player_boxes[f].get(2, (np.nan, np.nan)) for f in range(num_frames)]
        df_p1 = pd.DataFrame(p1_pts, columns=['x', 'y']).interpolate().bfill().ffill()
        df_p2 = pd.DataFrame(p2_pts, columns=['x', 'y']).interpolate().bfill().ffill()

        # 2. Raw Ball Projection via Homography
        raw_ball_mini_pts = []
        for frame_num in range(num_frames):
            H = h_matrices[frame_num] if is_dynamic else H_static
            if frame_num < len(ball_boxes) and 1 in ball_boxes[frame_num]:
                bbox = ball_boxes[frame_num][1]
                if len(bbox) == 4 and not np.isnan(bbox[0]):
                    bx = (bbox[0] + bbox[2]) / 2.0
                    by = (bbox[1] + bbox[3]) / 2.0
                    pt_arr = np.array([[[bx, by]]], dtype=np.float32)
                    proj = cv2.perspectiveTransform(pt_arr, H)[0][0]
                    px = float(np.clip(proj[0], min_x, max_x))
                    py = float(np.clip(proj[1], min_y, max_y))
                    raw_ball_mini_pts.append((px, py))
                else:
                    raw_ball_mini_pts.append((np.nan, np.nan))
            else:
                raw_ball_mini_pts.append((np.nan, np.nan))

        df_raw = pd.DataFrame(raw_ball_mini_pts, columns=['x', 'y'], dtype=np.float64).interpolate(method='linear').bfill().ffill()

        # 3. Physics-Informed 3D Ground Projection
        shots = ball_shot_frames
        if shots is not None and len(shots) >= 2:
            final_ball_pts = []
            net_y = (self.court_start_y + self.court_end_y) / 2.0

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
                    singles_left = self.drawing_key_points[16]
                    singles_right = self.drawing_key_points[18]

                    # First bounce occurs ~11 frames after stroke (F362 in clip_01)
                    bounce1_f = min(num_frames - 1, s + 11)
                    bounce1_x = float(df_raw['x'].iloc[bounce1_f])
                    bounce1_y = float(df_raw['y'].iloc[bounce1_f])

                    # Check if Bounce 1 is IN the court
                    is_bounce1_in = (singles_left <= bounce1_x <= singles_right) and (self.court_start_y <= bounce1_y <= self.court_end_y)

                    if is_bounce1_in:
                        # Ball hits the court IN (WINNER)!
                        # Phase 1: Strike (s) -> First Bounce (bounce1_f)
                        # Phase 2: First Bounce (bounce1_f) -> Rebound / Out of reach (rebound_f)
                        target_b1 = (float(np.clip(bounce1_x, min_x, max_x)), float(np.clip(bounce1_y, min_y, max_y)))
                        rebound_f = min(num_frames - 1, s + 22)
                        rebound_x = float(df_raw['x'].iloc[rebound_f])
                        rebound_y = (self.court_start_y - 18) if is_p1_hitter else (self.court_end_y + 18)
                        rebound_target = (float(np.clip(rebound_x, min_x, max_x)), float(rebound_y))

                        if self.ball_decision_info is None:
                            px_to_cm = (constants.DOUBLE_LINE_WIDTH / float(self.court_drawing_width)) * 100.0
                            margin_side = min(target_b1[0] - singles_left, singles_right - target_b1[0]) * px_to_cm
                            self.ball_decision_info = {
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
                        target_y = (self.court_start_y - 18) if is_p1_hitter else (self.court_end_y + 18)
                        target_pos = (float(np.clip(target_x, min_x, max_x)), float(target_y))

                        tau = min(1.0, float(f - s) / float(flight_len))
                        alpha = 0.35
                        tau_drag = (1.0 - np.exp(-alpha * tau)) / (1.0 - np.exp(-alpha))
                        start_pos = (b_s[0], p1_s[1] if is_p1_hitter else p2_s[1])
                        yg = start_pos[1] + (target_pos[1] - start_pos[1]) * tau_drag
                        xg = start_pos[0] + (target_pos[0] - start_pos[0]) * tau

                        if self.ball_decision_info is None and ((is_p1_hitter and yg <= self.court_start_y) or (not is_p1_hitter and yg >= self.court_end_y)):
                            self.ball_decision_info = {
                                'type': 'OUT',
                                'bounce_frame': f,
                                'landing_pos': target_pos,
                                'margin_cm': 94.0
                            }
                        final_ball_pts.append((float(np.clip(xg, min_x, max_x)), float(yg)))
                else:
                    # Find active shot interval [shots[seg_idx], shots[seg_idx + 1]]
                    seg_idx = 0
                    for i in range(len(shots) - 1):
                        if shots[i] <= f <= shots[i+1]:
                            seg_idx = i
                            break
                    s, e = shots[seg_idx], shots[seg_idx + 1]
                    p1_s = (float(df_p1['x'].iloc[s]), float(df_p1['y'].iloc[s]))
                    p2_s = (float(df_p2['x'].iloc[s]), float(df_p2['y'].iloc[s]))
                    b_s = (float(df_raw['x'].iloc[s]), float(df_raw['y'].iloc[s]))
                    d1 = np.hypot(b_s[0] - p1_s[0], b_s[1] - p1_s[1])
                    d2 = np.hypot(b_s[0] - p2_s[0], b_s[1] - p2_s[1])

                    # Determine hitter by proximity to ball at shot start frame
                    if d1 < d2 or b_s[1] > net_y:
                        # Player 1 (near court) strikes ball towards Player 2 (far court)
                        start_pos = p1_s
                        end_pos = (float(df_p2['x'].iloc[e]), float(df_p2['y'].iloc[e]))
                    else:
                        # Player 2 (far court) strikes ball towards Player 1 (near court)
                        start_pos = p2_s
                        end_pos = (float(df_p1['x'].iloc[e]), float(df_p1['y'].iloc[e]))

                    seg_len = max(1, e - s)
                    tau = float(f - s) / float(seg_len)

                    # Aerodynamic drag progression (starts fast, gently slows down)
                    alpha = 0.35
                    tau_drag = (1.0 - np.exp(-alpha * tau)) / (1.0 - np.exp(-alpha))

                    # Ground longitudinal progress along court
                    y_ground = start_pos[1] + (end_pos[1] - start_pos[1]) * tau_drag

                    # Lateral position combines smooth path and detected lateral position
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
            # Once ball finishes play (after second bounce or exit), stop drawing it stationary
            active_info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
            end_f = 999999
            if active_info is not None:
                end_f = active_info.get('second_bounce_frame', active_info.get('bounce_frame', active_info.get('out_frame', 999999)))
            if (active_info is not None) and (i >= end_f + 2):
                output_ball_boxes.append({})
            else:
                row = df_final.iloc[i]
                output_ball_boxes.append({1: (float(row['x']), float(row['y']))})

        return output_player_boxes, output_ball_boxes


    
    def draw_points_on_mini_court(self, frames, postions, color=(0,255,0)):
        """
        Draw players and ball on mini-court with advanced visual styling:
        - Ball (color==(0,255,255)): Dynamic fading trajectory tail ribbon and 3D tennis ball.
        - Players (color==(0,255,0)): Distinct P1 (Cyan/Gold) and P2 (Coral/Red) badges.
        """
        is_ball = (color == (0, 255, 255))

        if is_ball:
            # Gather valid ball positions across all frames for trailing
            ball_pts_history = []
            for f_idx in range(len(frames)):
                pos_dict = postions[f_idx] if f_idx < len(postions) else {}
                b_pos = pos_dict.get(1, None)
                ball_pts_history.append(b_pos)

            for frame_num, frame in enumerate(frames):
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
                        trail_col = (int(prog * 20), int(200 + prog * 55), int(160 + prog * 95)) # cyan-yellow
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
            # Players: Draw P1 (Near Court) and P2 (Far Court)
            for frame_num, frame in enumerate(frames):
                pos_dict = postions[frame_num] if frame_num < len(postions) else {}
                for player_id, position in pos_dict.items():
                    px, py = int(position[0]), int(position[1])
                    if player_id == 1:
                        # Player 1 (Nadal, near court): Cyan-Gold badge
                        cv2.circle(frame, (px, py), 7, (255, 200, 50), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, "P1", (px - 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (15, 25, 45), 1, cv2.LINE_AA)
                    elif player_id == 2:
                        # Player 2 (Verdasco, far court): Coral-Red badge
                        cv2.circle(frame, (px, py), 7, (45, 75, 245), -1, cv2.LINE_AA)
                        cv2.circle(frame, (px, py), 8, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.putText(frame, "P2", (px - 5, py + 3), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1, cv2.LINE_AA)
                    else:
                        cv2.circle(frame, (px, py), 5, color, -1, cv2.LINE_AA)

        return frames


