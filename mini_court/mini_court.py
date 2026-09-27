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

    def draw_court(self,frame):
        for i in range(0, len(self.drawing_key_points),2):
            x = int(self.drawing_key_points[i])
            y = int(self.drawing_key_points[i+1])
            cv2.circle(frame, (x,y),5, (0,0,255),-1)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.drawing_key_points[line[0]*2]), int(self.drawing_key_points[line[0]*2+1]))
            end_point = (int(self.drawing_key_points[line[1]*2]), int(self.drawing_key_points[line[1]*2+1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.drawing_key_points[0], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        net_end_point = (self.drawing_key_points[2], int((self.drawing_key_points[1] + self.drawing_key_points[5])/2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def draw_background_rectangle(self,frame):
        shapes = np.zeros_like(frame,np.uint8)
        # Draw the rectangle
        cv2.rectangle(shapes, (self.start_x, self.start_y), (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        out = frame.copy()
        alpha=0.5
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(frame, alpha, shapes, 1 - alpha, 0)[mask]

        return out

    def draw_mini_court(self,frames):
        output_frames = []
        for frame in frames:
            frame = self.draw_background_rectangle(frame)
            frame = self.draw_court(frame)
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

    def convert_bounding_boxes_to_mini_court_coordinates(self, player_boxes, ball_boxes, original_court_key_points):
        """
        Convert player and ball bounding boxes to mini-court coordinates using
        mathematically rigorous 2D Homography perspective transformation.
        Supports both static court keypoints (shape (28,)) and dynamic per-frame keypoints (shape (N, 28)).
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

        # 2. Project Ball with 3D Parabolic Height Compensation
        # Homography assumes Z=0 (court plane). When the ball is airborne (Z > 0),
        # camera perspective tilt projects it shifted forward (lower Y), causing the ball
        # to appear to "run ahead" of the video.
        # We compensate for this vertical parallax shift using a parabolic flight model.
        from scipy.signal import find_peaks

        raw_ball_mini_pts = []
        for frame_num in range(len(player_boxes)):
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

        df_mini = pd.DataFrame(raw_ball_mini_pts, columns=['x', 'y'], dtype=np.float64).interpolate(method='linear').bfill().ffill()

        # Find flight inflection points (hits and bounces)
        peaks, _ = find_peaks(df_mini['y'], distance=15, prominence=25)
        valleys, _ = find_peaks(-df_mini['y'], distance=15, prominence=25)
        extrema = sorted(list(set([0] + list(peaks) + list(valleys) + [len(df_mini) - 1])))

        # Apply parabolic height compensation between inflection points
        y_corrected = df_mini['y'].values.copy()
        for i in range(len(extrema) - 1):
            t_start = extrema[i]
            t_end = extrema[i + 1]
            seg_len = t_end - t_start
            if seg_len <= 3:
                continue
            for t in range(t_start, t_end):
                tau = (t - t_start) / seg_len
                # 35 px height compensation matches typical tennis ball altitude (~2.2m)
                offset = 35.0 * 4.0 * tau * (1.0 - tau)
                y_corrected[t] = float(np.clip(y_corrected[t] + offset, min_y, max_y))

        df_mini['y'] = y_corrected
        df_mini_smooth = df_mini.rolling(window=3, min_periods=1, center=True).mean()

        for i in range(len(df_mini_smooth)):
            row = df_mini_smooth.iloc[i]
            output_ball_boxes.append({1: (float(row['x']), float(row['y']))})

        return output_player_boxes, output_ball_boxes


    
    def draw_points_on_mini_court(self,frames,postions, color=(0,255,0)):
        for frame_num, frame in enumerate(frames):
            for _, position in postions[frame_num].items():
                x,y = position
                x= int(x)
                y= int(y)
                cv2.circle(frame, (x,y), 5, color, -1)
        return frames

