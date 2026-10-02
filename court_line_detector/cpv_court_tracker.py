import os
import pickle
import time
import cv2
import numpy as np


def estimate_court_homography(previous_points, current_points, reprojection_threshold=3.0):
    """Estimate full perspective camera motion and report its RANSAC inlier ratio."""
    previous_points = np.asarray(previous_points, dtype=np.float32).reshape((-1, 2))
    current_points = np.asarray(current_points, dtype=np.float32).reshape((-1, 2))
    if len(previous_points) < 8 or len(current_points) != len(previous_points):
        return None, 0.0

    matrix, inliers = cv2.findHomography(
        previous_points,
        current_points,
        method=cv2.RANSAC,
        ransacReprojThreshold=reprojection_threshold,
    )
    if matrix is None or inliers is None:
        return None, 0.0
    return matrix, float(np.mean(inliers))


def track_court_keypoints_cpv(
    video_frames,
    initial_keypoints=None,
    detector=None,
    stub_path=None,
    read_from_stub=True,
    redetect_interval=60,
):
    """
    [HYBRID: TRACKNET CORRECTION + CPV OPTICAL FLOW TRACKING]
    
    Thuật toán tracking vạch sân hybrid TrackNet + Computer Vision (CPV):
    1. Frame 0: Nhận diện chính xác 14 keypoint bằng mô hình AI + Line Refinement.
    2. Từ Frame 1 đến hết: Sử dụng Lucas-Kanade Optical Flow hai chiều kết hợp
       Homography RANSAC trên các điểm đặc trưng tĩnh của mặt sân.
    
    Ưu điểm:
    - Bám dính chính xác từng pixel theo thời gian thực (Zero Lag, không bị trễ do lọc trung bình).
    - Giảm drift khi camera pan, tilt hoặc zoom bằng homography và tái-detect thích nghi.
    - Miễn nhiễm với biển quảng cáo, khán đài và bóng người di chuyển nhờ RANSAC Outlier Rejection.
    - Duy trì tốc độ xấp xỉ realtime trên GPU; tốc độ phụ thuộc số lần TrackNet tái-detect.
    - Hoàn toàn độc lập, dễ dàng bật/tắt hoặc gỡ bỏ.

    Parameters:
    -----------
    video_frames : iterable of np.ndarray
        Các frame ảnh BGR của video theo thứ tự thời gian.
    initial_keypoints : np.ndarray, optional
        14 toạ độ keypoints (28 số) của frame 0. Nếu None, sẽ dùng `detector.predict(video_frames[0])`.
    detector : CourtLineDetector, optional
        Đối tượng detector để suy luận frame 0 nếu chưa có `initial_keypoints`.
    stub_path : str, optional
        Đường dẫn file cache (.pkl). Nếu đã có, load ngay lập tức trong 0.01s.

    Returns:
    --------
    np.ndarray of shape (số frame, 28)
        Mảng toạ độ 14 keypoints cho từng frame tương ứng.
    """
    # 1. Kiểm tra cache
    if read_from_stub and stub_path and os.path.exists(stub_path):
        print(f"[CPV_CourtTracker] Loading cached keypoints from {stub_path}")
        with open(stub_path, "rb") as f:
            return pickle.load(f)

    frames = iter(video_frames)
    first_frame = next(frames, None)
    if first_frame is None:
        return np.array([])

    print("[CPV_CourtTracker] Running Homography Tracking with TrackNet Correction...")
    t0 = time.time()

    # 2. Khởi tạo Frame 0
    if initial_keypoints is None:
        if detector is None:
            raise ValueError("[CPV_CourtTracker] Either initial_keypoints or detector must be provided.")
        initial_keypoints = detector.predict(first_frame)

    kps_0 = np.array(initial_keypoints, dtype=np.float32)
    current_pts = np.array([(kps_0[2 * i], kps_0[2 * i + 1]) for i in range(14)], dtype=np.float32)

    all_kps = [kps_0.copy()]

    # 3. Tạo mặt nạ vùng sân (Court Mask) để chỉ bắt đặc trưng mặt sân, loại trừ khán đài & biển quảng cáo
    h, w = first_frame.shape[:2]
    court_mask = np.zeros((h, w), dtype=np.uint8)
    court_mask[int(h * 0.25):int(h * 0.95), int(w * 0.10):int(w * 0.90)] = 255

    prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)

    # Khởi tạo điểm đặc trưng ban đầu bằng Shi-Tomasi (Good Features to Track)
    prev_pts_features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=250,
        qualityLevel=0.03,
        minDistance=15,
        mask=court_mask
    )

    # 4. Tracking liên tục qua từng frame bằng Lucas-Kanade Optical Flow + RANSAC Homography
    for f_idx, frame in enumerate(frames, start=1):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform_applied = False
        camera_motion = 0.0

        if prev_pts_features is not None and len(prev_pts_features) >= 10:
            next_pts_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts_features, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            if next_pts_features is None or status is None:
                good_prev = np.empty((0, 2), dtype=np.float32)
                good_next = np.empty((0, 2), dtype=np.float32)
            else:
                back_pts_features, back_status, _ = cv2.calcOpticalFlowPyrLK(
                    curr_gray, prev_gray, next_pts_features, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )
                valid = status.reshape(-1) == 1
                if back_pts_features is not None and back_status is not None:
                    forward_backward_error = np.linalg.norm(
                        prev_pts_features.reshape((-1, 2))
                        - back_pts_features.reshape((-1, 2)),
                        axis=1,
                    )
                    valid &= back_status.reshape(-1) == 1
                    valid &= forward_backward_error <= 1.5
                else:
                    valid[:] = False
                good_prev = prev_pts_features.reshape((-1, 2))[valid]
                good_next = next_pts_features.reshape((-1, 2))[valid]

            if len(good_prev) >= 8:
                matrix, inlier_ratio = estimate_court_homography(good_prev, good_next)
                if matrix is not None and inlier_ratio >= 0.5:
                    transformed_points = cv2.perspectiveTransform(
                        current_pts.reshape((-1, 1, 2)), matrix
                    ).reshape((-1, 2))
                    camera_motion = float(
                        np.median(np.linalg.norm(transformed_points - current_pts, axis=1))
                    )
                    current_pts = transformed_points
                    transform_applied = True

            # Bổ sung lại feature points nếu số lượng điểm bám bị giảm
            if len(good_next) < 80:
                prev_pts_features = cv2.goodFeaturesToTrack(
                    curr_gray, maxCorners=250, qualityLevel=0.03, minDistance=15, mask=court_mask
                )
            else:
                prev_pts_features = good_next.reshape(-1, 1, 2)
        else:
            prev_pts_features = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=250, qualityLevel=0.03, minDistance=15, mask=court_mask
            )

        periodic_redetection = redetect_interval and f_idx % redetect_interval == 0
        recovery_redetection = not transform_applied and f_idx % 12 == 0
        motion_redetection = camera_motion >= 2.5
        if detector is not None and (
            periodic_redetection or recovery_redetection or motion_redetection
        ):
            try:
                detected_points = np.asarray(detector.predict(frame), dtype=np.float32).reshape((-1, 2))
                if detected_points.shape == (14, 2) and np.isfinite(detected_points).all():
                    current_pts = detected_points
                    prev_pts_features = cv2.goodFeaturesToTrack(
                        curr_gray,
                        maxCorners=250,
                        qualityLevel=0.03,
                        minDistance=15,
                        mask=court_mask,
                    )
            except RuntimeError:
                pass

        prev_gray = curr_gray

        # Lưu toạ độ 14 keypoint cho frame hiện tại
        kps_f = np.zeros(28, dtype=np.float32)
        for i in range(14):
            kps_f[2 * i] = current_pts[i, 0]
            kps_f[2 * i + 1] = current_pts[i, 1]
        all_kps.append(kps_f)

    elapsed = time.time() - t0
    final_keypoints = np.array(all_kps, dtype=np.float32)
    total_frames = len(all_kps)
    print(f"[CPV_CourtTracker] Finished CPV Tracking {total_frames} frames in {elapsed:.3f}s ({total_frames / max(1e-4, elapsed):.1f} FPS)!")

    # 5. Lưu cache
    if stub_path:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump(final_keypoints, f)
        print(f"[CPV_CourtTracker] Saved keypoints to {stub_path}")

    return final_keypoints
