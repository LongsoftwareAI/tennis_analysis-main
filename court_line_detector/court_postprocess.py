import cv2
import numpy as np


COURT_REFERENCE_KEYPOINTS = np.array(
    [
        (286, 561), (1379, 561),
        (286, 2935), (1379, 2935),
        (423, 561), (423, 2935),
        (1242, 561), (1242, 2935),
        (423, 1110), (1242, 1110),
        (423, 2386), (1242, 2386),
        (832, 1110), (832, 2386),
    ],
    dtype=np.float32,
)

_HOMOGRAPHY_CONFIGURATIONS = (
    (0, 1, 2, 3),
    (4, 6, 5, 7),
    (4, 1, 5, 3),
    (0, 6, 2, 7),
    (8, 9, 10, 11),
    (8, 9, 5, 7),
    (4, 6, 10, 11),
    (6, 1, 7, 3),
    (0, 4, 2, 5),
    (8, 12, 10, 13),
    (12, 9, 13, 11),
    (10, 11, 5, 7),
)


def extract_keypoint(heatmap, scale_x=1.0, scale_y=1.0, threshold=170):
    """Locate a heatmap blob and map it from 640x360 space to the source frame."""
    binary = cv2.threshold(heatmap, threshold, 255, cv2.THRESH_BINARY)[1]
    circles = cv2.HoughCircles(
        binary,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=20,
        param1=50,
        param2=2,
        minRadius=10,
        maxRadius=25,
    )

    if circles is not None:
        x_coord, y_coord = circles[0, 0, :2]
    else:
        component_count, _, stats, centroids = cv2.connectedComponentsWithStats(binary)
        if component_count <= 1:
            return None
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        x_coord, y_coord = centroids[largest_component]

    return np.array((x_coord * scale_x, y_coord * scale_y), dtype=np.float32)


def _line_intersection(first, second):
    x1, y1, x2, y2 = first
    x3, y3, x4, y4 = second
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denominator) < 1e-6:
        return None

    determinant_1 = x1 * y2 - y1 * x2
    determinant_2 = x3 * y4 - y3 * x4
    x_coord = (determinant_1 * (x3 - x4) - (x1 - x2) * determinant_2) / denominator
    y_coord = (determinant_1 * (y3 - y4) - (y1 - y2) * determinant_2) / denominator
    return x_coord, y_coord


def refine_keypoint(image, point, crop_size=40):
    """Snap a coarse prediction to the strongest local court-line intersection."""
    x_coord, y_coord = (int(round(value)) for value in point)
    height, width = image.shape[:2]
    x_min, x_max = max(0, x_coord - crop_size), min(width, x_coord + crop_size)
    y_min, y_max = max(0, y_coord - crop_size), min(height, y_coord + crop_size)
    crop = image[y_min:y_max, x_min:x_max]
    if crop.size == 0:
        return point

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    binary = cv2.threshold(gray, 155, 255, cv2.THRESH_BINARY)[1]
    detected = cv2.HoughLinesP(binary, 1, np.pi / 180, 30, minLineLength=10, maxLineGap=30)
    if detected is None or len(detected) < 2:
        return point

    lines = np.asarray(detected).reshape((-1, 4))
    best_intersection = None
    best_score = 0.0
    for index, first in enumerate(lines):
        first_vector = first[2:] - first[:2]
        first_length = float(np.linalg.norm(first_vector))
        for second in lines[index + 1:]:
            second_vector = second[2:] - second[:2]
            second_length = float(np.linalg.norm(second_vector))
            if first_length == 0.0 or second_length == 0.0:
                continue
            cosine = abs(float(np.dot(first_vector, second_vector) / (first_length * second_length)))
            if cosine > 0.94:
                continue
            intersection = _line_intersection(first, second)
            if intersection is None:
                continue
            local_x, local_y = intersection
            if 0 <= local_x < crop.shape[1] and 0 <= local_y < crop.shape[0]:
                score = first_length + second_length
                if score > best_score:
                    best_intersection = (x_min + local_x, y_min + local_y)
                    best_score = score

    if best_intersection is None:
        return point
    return np.array(best_intersection, dtype=np.float32)


def reconstruct_keypoints(points):
    """Use the best valid four-point court configuration to rebuild all 14 points."""
    valid = np.array([point is not None for point in points], dtype=bool)
    if valid.sum() < 4:
        return None

    best_keypoints = None
    best_error = np.inf
    reference = COURT_REFERENCE_KEYPOINTS.reshape((-1, 1, 2))

    for indices in _HOMOGRAPHY_CONFIGURATIONS:
        if not all(valid[index] for index in indices):
            continue
        source = COURT_REFERENCE_KEYPOINTS[list(indices)]
        destination = np.array([points[index] for index in indices], dtype=np.float32)
        matrix = cv2.getPerspectiveTransform(source, destination)
        transformed = cv2.perspectiveTransform(reference, matrix).reshape((-1, 2))

        check_indices = [index for index in range(14) if valid[index] and index not in indices]
        if check_indices:
            observed = np.array([points[index] for index in check_indices], dtype=np.float32)
            error = float(np.mean(np.linalg.norm(observed - transformed[check_indices], axis=1)))
        else:
            error = 0.0

        if error < best_error:
            best_keypoints = transformed
            best_error = error

    return best_keypoints
