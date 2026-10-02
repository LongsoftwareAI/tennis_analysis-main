from .geometry import MiniCourtGeometry
from .drawer import MiniCourtDrawer
from .projector import MiniCourtProjector

class MiniCourt:
    """
    Facade orchestrator for Mini-Court 2D Radar system.
    Coordinates geometry, homography & physics projection, and graphical rendering
    while maintaining full backward compatibility with external modules.
    """
    def __init__(self, frame):
        self.geometry = MiniCourtGeometry(frame)
        self.drawer = MiniCourtDrawer(self.geometry)
        self.projector = MiniCourtProjector(self.geometry)

        # Expose geometry properties for direct access
        self.drawing_rectangle_width = self.geometry.drawing_rectangle_width
        self.drawing_rectangle_height = self.geometry.drawing_rectangle_height
        self.buffer = self.geometry.buffer
        self.padding_court = self.geometry.padding_court

        self.start_x = self.geometry.start_x
        self.start_y = self.geometry.start_y
        self.end_x = self.geometry.end_x
        self.end_y = self.geometry.end_y
        self.court_start_x = self.geometry.court_start_x
        self.court_start_y = self.geometry.court_start_y
        self.court_end_x = self.geometry.court_end_x
        self.court_end_y = self.geometry.court_end_y
        self.court_drawing_width = self.geometry.court_drawing_width
        self.drawing_key_points = self.geometry.drawing_key_points
        self.lines = self.geometry.lines

        self.ball_out_info = None
        self.ball_decision_info = None
        self.all_bounces = []

    def set_referee_decision(self, decision_info, all_bounces=None):
        """Register the referee decision info and all detected rally bounces for mini-court visualization."""
        self.ball_decision_info = decision_info
        if all_bounces is not None:
            self.all_bounces = all_bounces

    def convert_meters_to_pixels(self, meters):
        """Convert real-world distance in meters to mini-court pixel distance."""
        return self.geometry.convert_meters_to_pixels(meters)

    def get_width_of_mini_court(self):
        return self.geometry.get_width_of_mini_court()

    def get_start_point_of_mini_court(self):
        return self.geometry.get_start_point_of_mini_court()

    def get_court_drawing_keypoints(self):
        return self.geometry.get_court_drawing_keypoints()

    def get_mini_court_coordinates(self,
                                   object_position,
                                   closest_key_point, 
                                   closest_key_point_index, 
                                   player_height_in_pixels,
                                   player_height_in_meters):
        return self.geometry.get_mini_court_coordinates(
            object_position,
            closest_key_point,
            closest_key_point_index,
            player_height_in_pixels,
            player_height_in_meters
        )

    def convert_bounding_boxes_to_mini_court_coordinates(
        self,
        player_boxes,
        ball_boxes,
        original_court_key_points,
        ball_shot_frames=None,
        all_bounces=None
    ):
        """Convert player and ball bounding boxes using homography and aerodynamic ground projection."""
        existing_info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
        bounces = all_bounces if all_bounces is not None else self.all_bounces
        player_pts, ball_pts, dec_info = self.projector.convert_bounding_boxes_to_mini_court_coordinates(
            player_boxes,
            ball_boxes,
            original_court_key_points,
            ball_shot_frames=ball_shot_frames,
            existing_decision_info=existing_info,
            all_bounces=bounces
        )
        if self.ball_decision_info is None and dec_info is not None:
            self.ball_decision_info = dec_info

        return player_pts, ball_pts

    def draw_background_rectangle(self, frame):
        return self.drawer.draw_background_rectangle(frame)

    def draw_court(self, frame):
        return self.drawer.draw_court(frame)

    def draw_bounce_impacts(self, frame, frame_num):
        active_info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
        return self.drawer.draw_bounce_impacts(frame, frame_num, all_bounces=self.all_bounces, decision_info=active_info)

    def draw_decision_indicator(self, frame):
        active_info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
        return self.drawer.draw_decision_indicator(frame, active_info)

    def draw_mini_court(self, frames, start_frame=0):
        active_info = self.ball_decision_info if self.ball_decision_info is not None else self.ball_out_info
        return self.drawer.draw_mini_court(frames, all_bounces=self.all_bounces, decision_info=active_info,
                                           start_frame=start_frame)

    def draw_points_on_mini_court(self, frames, positions, color=(0, 255, 0),
                                  start_frame=0, ball_history=None):
        return self.drawer.draw_points_on_mini_court(frames, positions, color=color,
                                                     start_frame=start_frame, ball_history=ball_history)
