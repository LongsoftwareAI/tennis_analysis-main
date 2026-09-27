import sys
sys.path.append('../')
import constants
from utils import (
    convert_meters_to_pixel_distance,
    convert_pixel_distance_to_meters,
    measure_xy_distance
)

class MiniCourtGeometry:
    """
    Manages coordinate transformations, keypoints, dimensions,
    and court lines for the 2D Mini-Court radar.
    """
    def __init__(self, frame, drawing_rectangle_width=250, drawing_rectangle_height=500, buffer=50, padding_court=20):
        self.drawing_rectangle_width = drawing_rectangle_width
        self.drawing_rectangle_height = drawing_rectangle_height
        self.buffer = buffer
        self.padding_court = padding_court

        self.set_canvas_background_box_position(frame)
        self.set_mini_court_position()
        self.set_court_drawing_key_points()
        self.set_court_lines()

    def set_canvas_background_box_position(self, frame):
        """Calculate the outer radar canvas bounding box within video frame."""
        shape = frame.shape if hasattr(frame, 'shape') else frame
        height, width = shape[0], shape[1]

        self.end_x = width - self.buffer
        self.end_y = self.buffer + self.drawing_rectangle_height
        self.start_x = self.end_x - self.drawing_rectangle_width
        self.start_y = self.end_y - self.drawing_rectangle_height

    def set_mini_court_position(self):
        """Set court boundary inside the canvas with padding."""
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_drawing_width = self.court_end_x - self.court_start_x

    def convert_meters_to_pixels(self, meters):
        """Convert real-world distance in meters to mini-court pixel distance."""
        return convert_meters_to_pixel_distance(
            meters,
            constants.DOUBLE_LINE_WIDTH,
            self.court_drawing_width
        )

    def set_court_drawing_key_points(self):
        """
        Calculate all 14 standard tennis court keypoints (28 coordinate elements: x0, y0, x1, y1, ...):
        - 0, 1: Far court baseline corners (doubles)
        - 2, 3: Near court baseline corners (doubles)
        - 4, 5: Far and near doubles ally left line
        - 6, 7: Far and near doubles ally right line
        - 8, 9: Far court service line
        - 10, 11: Near court service line
        - 12, 13: Far and near center service line T-junctions
        """
        drawing_key_points = [0] * 28

        # point 0: Top-left outer baseline corner
        drawing_key_points[0], drawing_key_points[1] = int(self.court_start_x), int(self.court_start_y)
        # point 1: Top-right outer baseline corner
        drawing_key_points[2], drawing_key_points[3] = int(self.court_end_x), int(self.court_start_y)
        # point 2: Bottom-left outer baseline corner
        drawing_key_points[4] = int(self.court_start_x)
        drawing_key_points[5] = self.court_start_y + self.convert_meters_to_pixels(constants.HALF_COURT_LINE_HEIGHT * 2)
        # point 3: Bottom-right outer baseline corner
        drawing_key_points[6] = drawing_key_points[0] + self.court_drawing_width
        drawing_key_points[7] = drawing_key_points[5]
        # point 4: Top-left singles sideline junction
        drawing_key_points[8] = drawing_key_points[0] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[9] = drawing_key_points[1]
        # point 5: Bottom-left singles sideline junction
        drawing_key_points[10] = drawing_key_points[4] + self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[11] = drawing_key_points[5]
        # point 6: Top-right singles sideline junction
        drawing_key_points[12] = drawing_key_points[2] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[13] = drawing_key_points[3]
        # point 7: Bottom-right singles sideline junction
        drawing_key_points[14] = drawing_key_points[6] - self.convert_meters_to_pixels(constants.DOUBLE_ALLY_DIFFERENCE)
        drawing_key_points[15] = drawing_key_points[7]
        # point 8: Far court service line left junction
        drawing_key_points[16] = drawing_key_points[8]
        drawing_key_points[17] = drawing_key_points[9] + self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 9: Far court service line right junction
        drawing_key_points[18] = drawing_key_points[16] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[19] = drawing_key_points[17]
        # point 10: Near court service line left junction
        drawing_key_points[20] = drawing_key_points[10]
        drawing_key_points[21] = drawing_key_points[11] - self.convert_meters_to_pixels(constants.NO_MANS_LAND_HEIGHT)
        # point 11: Near court service line right junction
        drawing_key_points[22] = drawing_key_points[20] + self.convert_meters_to_pixels(constants.SINGLE_LINE_WIDTH)
        drawing_key_points[23] = drawing_key_points[21]
        # point 12: Far court center service T-junction
        drawing_key_points[24] = int((drawing_key_points[16] + drawing_key_points[18]) / 2)
        drawing_key_points[25] = drawing_key_points[17]
        # point 13: Near court center service T-junction
        drawing_key_points[26] = int((drawing_key_points[20] + drawing_key_points[22]) / 2)
        drawing_key_points[27] = drawing_key_points[21]

        self.drawing_key_points = drawing_key_points

    def set_court_lines(self):
        """Define pairs of keypoint indices that connect to form court lines."""
        self.lines = [
            (0, 2),
            (4, 5),
            (6, 7),
            (1, 3),
            
            (0, 1),
            (8, 9),
            (10, 11),
            (10, 11),
            (2, 3)
        ]

    def get_start_point_of_mini_court(self):
        return (self.court_start_x, self.court_start_y)

    def get_width_of_mini_court(self):
        return self.court_drawing_width

    def get_court_drawing_keypoints(self):
        return self.drawing_key_points

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters):
        """Convert a detected object position relative to nearest court keypoint into mini-court space."""
        distance_from_keypoint_x_pixels, distance_from_keypoint_y_pixels = measure_xy_distance(object_position, closest_key_point)

        distance_from_keypoint_x_meters = convert_pixel_distance_to_meters(
            distance_from_keypoint_x_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )
        distance_from_keypoint_y_meters = convert_pixel_distance_to_meters(
            distance_from_keypoint_y_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )

        mini_court_x_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_x_meters)
        mini_court_y_distance_pixels = self.convert_meters_to_pixels(distance_from_keypoint_y_meters)
        closest_mini_court_keypoint = (
            self.drawing_key_points[closest_key_point_index * 2],
            self.drawing_key_points[closest_key_point_index * 2 + 1]
        )

        mini_court_player_position = (
            closest_mini_court_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_court_keypoint[1] + mini_court_y_distance_pixels
        )

        return mini_court_player_position
