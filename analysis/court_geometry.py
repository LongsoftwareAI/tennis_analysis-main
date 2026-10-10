import numpy as np
import constants

class CourtGeometry:
    """
    Manages tennis court metric coordinates and spatial transformations:
    - Standard ATP court dimensions (meters)
    - Mini-Court pixel <-> real-world court metric conversions
    - Spatial depth, side, and zone classifications
    """
    COURT_LENGTH = constants.HALF_COURT_LINE_HEIGHT * 2.0  # 23.77 m
    DOUBLES_WIDTH = constants.DOUBLE_LINE_WIDTH             # 10.97 m
    SINGLES_WIDTH = constants.SINGLE_LINE_WIDTH             # 8.23 m
    SERVICE_LINE_DIST = 6.40                                # 6.40 m from net
    BASELINE_DIST = 11.885                                  # 11.885 m from net
    DEEP_ZONE_DIST = 10.08                                  # 1.8 m from baseline
    ALLEY_DIFF = constants.DOUBLE_ALLY_DIFFERENCE           # 1.37 m

    def __init__(self):
        self.court_length = self.COURT_LENGTH
        self.court_width_doubles = self.DOUBLES_WIDTH
        self.court_width_singles = self.SINGLES_WIDTH
        self.service_line_dist = self.SERVICE_LINE_DIST
        self.baseline_dist = self.BASELINE_DIST
        self.deep_zone_dist = self.DEEP_ZONE_DIST
        self.alley_diff = self.ALLEY_DIFF

    def mini_px_to_court_meters(self, mini_court, px, py):
        """
        Chuyển đổi tọa độ pixel trên Mini-Court sang hệ mét thực tế:
          X: [-5.485, +5.485] m (0 là tâm vạch giữa)
          Y: [-11.885, +11.885] m (0 là lưới; < 0 là Player 2 sân xa, > 0 là Player 1 sân gần)
        """
        if mini_court is None or px is None or py is None or np.isnan(px) or np.isnan(py):
            return 0.0, 0.0

        cx = (mini_court.court_start_x + mini_court.court_end_x) / 2.0
        cy = (mini_court.court_start_y + mini_court.court_end_y) / 2.0
        court_w_px = max(1.0, float(mini_court.court_drawing_width))
        court_h_px = max(1.0, float(mini_court.court_end_y - mini_court.court_start_y))

        scale_x = self.DOUBLES_WIDTH / court_w_px
        scale_y = self.COURT_LENGTH / court_h_px

        x_m = (float(px) - cx) * scale_x
        y_m = (float(py) - cy) * scale_y
        return x_m, y_m

    def court_meters_to_mini_px(self, mini_court, x_m, y_m):
        """Chuyển đổi ngược từ hệ mét thực tế sang tọa độ pixel Mini-Court."""
        if mini_court is None:
            return 0.0, 0.0

        cx = (mini_court.court_start_x + mini_court.court_end_x) / 2.0
        cy = (mini_court.court_start_y + mini_court.court_end_y) / 2.0
        court_w_px = max(1.0, float(mini_court.court_drawing_width))
        court_h_px = max(1.0, float(mini_court.court_end_y - mini_court.court_start_y))

        scale_x = court_w_px / self.DOUBLES_WIDTH
        scale_y = court_h_px / self.COURT_LENGTH

        px = cx + (x_m * scale_x)
        py = cy + (y_m * scale_y)
        return px, py

    def classify_zone(self, y_m):
        """Phân loại 3 vùng sân chiến thuật: Baseline vs No-Man's Land vs Attack Zone."""
        abs_y = abs(y_m)
        if abs_y >= 10.5:
            return "Baseline (Day san)"
        elif abs_y >= 6.4:
            return "No-Man's Land (Giua san)"
        else:
            return "Attack Zone (Tan cong luoi)"

    def classify_depth(self, y_m):
        """Phân loại độ sâu bóng nảy: Deep vs Medium vs Short."""
        abs_y = abs(y_m)
        if abs_y >= self.DEEP_ZONE_DIST:
            return "Deep (Bong sau)"
        elif abs_y >= self.SERVICE_LINE_DIST:
            return "Medium (Giua san)"
        else:
            return "Short (Bong ngan)"
