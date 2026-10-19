import numpy as np
import constants


class CourtGeometry:
    """
    Manages tennis court metric coordinates and spatial transformations:
    - Standard ATP court dimensions (meters)
    - Mini-Court pixel <-> real-world court metric conversions
    - Spatial depth, side, and zone classifications
    """
    COURT_LENGTH = 23.77                                    # m (constants.HALF_COURT_LINE_HEIGHT * 2 = 23.76 bi lech 1cm)
    DOUBLES_WIDTH = constants.DOUBLE_LINE_WIDTH             # 10.97 m
    SINGLES_WIDTH = constants.SINGLE_LINE_WIDTH             # 8.23 m
    SERVICE_LINE_DIST = 6.40                                # 6.40 m from net
    BASELINE_DIST = COURT_LENGTH / 2.0                      # 11.885 m from net
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

    # ------------------------------------------------------------------
    # Khung toa do mini-court
    # ------------------------------------------------------------------
    def _court_frame(self, mini_court):
        """
        Tra ve (cx, cy, px_per_m_x, px_per_m_y) cua san VE THAT tren mini-court.

        Truoc day cy/scale_y lay tu court_start_y..court_end_y (bao gom padding),
        trong khi san that duoc ve tu keypoint 0 (baseline xa) den keypoint 2 (baseline gan)
        va ngan hon ~5px => lech ~1% theo truc y va lech tam ~0.14 m.
        Nen uu tien dung chinh drawing_key_points de khop voi cach ve.
        """
        cx = (mini_court.court_start_x + mini_court.court_end_x) / 2.0
        court_w_px = max(1.0, float(mini_court.court_drawing_width))
        px_per_m_x = court_w_px / self.DOUBLES_WIDTH

        kp = getattr(mini_court, "drawing_key_points", None)
        if kp is not None and len(kp) >= 6 and float(kp[5]) > float(kp[1]):
            far_y, near_y = float(kp[1]), float(kp[5])      # baseline xa / gan
        else:
            far_y, near_y = float(mini_court.court_start_y), float(mini_court.court_end_y)

        cy = (far_y + near_y) / 2.0
        px_per_m_y = max(1.0, near_y - far_y) / self.COURT_LENGTH
        return cx, cy, px_per_m_x, px_per_m_y

    def mini_px_to_court_meters(self, mini_court, px, py, nan_if_invalid=False):
        """
        Chuyen toa do pixel tren Mini-Court sang he met thuc te:
          X: [-5.485, +5.485] m (0 la tam vach giua)
          Y: [-11.885, +11.885] m (0 la luoi; < 0 la Player 2 san xa, > 0 la Player 1 san gan)

        Dau vao khong hop le: mac dinh tra (0, 0) de tuong thich nguoc;
        dat nan_if_invalid=True de nhan (nan, nan) thay vi mot diem gia tai luoi.
        """
        if mini_court is None or px is None or py is None or np.isnan(px) or np.isnan(py):
            return (np.nan, np.nan) if nan_if_invalid else (0.0, 0.0)

        cx, cy, ppm_x, ppm_y = self._court_frame(mini_court)
        x_m = (float(px) - cx) / ppm_x
        y_m = (float(py) - cy) / ppm_y
        return x_m, y_m

    def court_meters_to_mini_px(self, mini_court, x_m, y_m):
        """Chuyen nguoc tu he met thuc te sang toa do pixel Mini-Court."""
        if mini_court is None:
            return 0.0, 0.0

        cx, cy, ppm_x, ppm_y = self._court_frame(mini_court)
        px = cx + x_m * ppm_x
        py = cy + y_m * ppm_y
        return px, py

    # ------------------------------------------------------------------
    # Phan loai
    # ------------------------------------------------------------------
    def classify_zone(self, y_m):
        """Phan loai 3 vung san chien thuat: Baseline vs No-Man's Land vs Attack Zone."""
        if y_m is None or np.isnan(y_m):
            return "N/A"
        abs_y = abs(y_m)
        if abs_y >= 10.5:
            return "Baseline (Day san)"
        elif abs_y >= 6.4:
            return "No-Man's Land (Giua san)"
        else:
            return "Attack Zone (Tan cong luoi)"

    def classify_depth(self, y_m):
        """Phan loai do sau bong nay: Deep vs Medium vs Short."""
        abs_y = abs(y_m)
        if abs_y >= self.DEEP_ZONE_DIST:
            return "Deep (Bong sau)"
        elif abs_y >= self.SERVICE_LINE_DIST:
            return "Medium (Giua san)"
        else:
            return "Short (Bong ngan)"