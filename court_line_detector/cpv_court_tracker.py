import os
import pickle
import time
import cv2
import numpy as np


COURT_LINE_SEGMENTS = np.array(
    ((0, 1), (0, 2), (1, 3), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13))
)


def court_line_contrast(gray, points):
    """Measure how well projected court lines align with bright image lines."""
    starts = points[COURT_LINE_SEGMENTS[:, 0]]
    directions = points[COURT_LINE_SEGMENTS[:, 1]] - starts
    lengths = np.linalg.norm(directions, axis=1)
    if np.any(lengths < 1.0):
        return -np.inf
    normals = np.stack((-directions[:, 1], directions[:, 0]), axis=1) / lengths[:, None]
    centers = starts[:, None, :] + directions[:, None, :] * np.linspace(.08, .92, 16)[None, :, None]
    sides = normals[:, None, :] * 5.0
    height, width = gray.shape
    valid = (
        (centers[:, :, 0] >= 6) & (centers[:, :, 0] < width - 6)
        & (centers[:, :, 1] >= 6) & (centers[:, :, 1] < height - 6)
    )
    if np.count_nonzero(valid) < 40:
        return -np.inf

    def sample(locations):
        return cv2.remap(
            gray, locations[:, :, 0].astype(np.float32),
            locations[:, :, 1].astype(np.float32), cv2.INTER_LINEAR,
        ).astype(np.float32)

    contrast = sample(centers) - (sample(centers - sides) + sample(centers + sides)) / 2.0
    return float(np.mean(contrast[valid]))


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
    [HYBRID: TRACKNET CORRECTION + KEYFRAME OPTICAL FLOW TRACKING]
    
    Thuật toán tracking vạch sân hybrid TrackNet + Computer Vision (CPV):
    1. Frame 0: Nhận diện chính xác 14 keypoint bằng mô hình AI + Line Refinement.
    2. Từ Frame 1 đến hết: Sử dụng Lucas-Kanade Optical Flow hai chiều kết hợp
       Homography RANSAC so với frame mốc gần nhất.
    
    Ưu điểm:
    - Giữ điểm ổn định khi nền sân đứng yên nhưng vẫn theo được pan/tilt/zoom chậm.
    - Chỉ nhận tái-detect khi vạch sân trong ảnh hỗ trợ tọa độ mới.

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
    buffered_frames = []
    # Buffer up to 40 frames (~1.3s) for robust initial anchor screening
    for _ in range(40):
        f = next(frames, None)
        if f is None:
            break
        buffered_frames.append(f)

    if not buffered_frames:
        return np.array([])

    first_frame = buffered_frames[0]
    h, w = first_frame.shape[:2]
    court_mask = np.zeros((h, w), dtype=np.uint8)
    court_mask[int(h * 0.25):int(h * 0.95), int(w * 0.10):int(w * 0.90)] = 255

    print("[CPV_CourtTracker] Running Bi-directional Multi-frame Anchor Screening & Homography Tracking...")
    t0 = time.time()

    # 2. Khởi tạo Keyframe chuẩn xác nhất trong buffer
    best_k = 0
    best_pts = None
    best_score = -np.inf

    if initial_keypoints is not None:
        kps_0 = np.array(initial_keypoints, dtype=np.float32)
        best_pts = np.array([(kps_0[2 * i], kps_0[2 * i + 1]) for i in range(14)], dtype=np.float32)
        best_score = court_line_contrast(cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY), best_pts)
    elif detector is not None:
        sample_indices = [i for i in range(0, len(buffered_frames), 5)]
        for idx in sample_indices:
            try:
                cand_pts = np.asarray(detector.predict(buffered_frames[idx]), dtype=np.float32).reshape((-1, 2))
                cand_gray = cv2.cvtColor(buffered_frames[idx], cv2.COLOR_BGR2GRAY)
                cand_score = court_line_contrast(cand_gray, cand_pts)
                if cand_score > best_score:
                    best_score = cand_score
                    best_k = idx
                    best_pts = cand_pts
            except Exception:
                continue

        if best_pts is None:
            kps_0 = detector.predict(first_frame)
            best_pts = np.asarray(kps_0, dtype=np.float32).reshape((-1, 2))
            best_k = 0

    print(f"[CPV_CourtTracker] Multi-frame Anchor Screening: Locked optimal Anchor at F{best_k} (contrast = {best_score:.2f}).")

    # Bi-directional tracking on buffer
    kps_buffer = [None] * len(buffered_frames)
    kps_buffer[best_k] = best_pts.copy()

    # Backwards tracking from best_k down to Frame 0
    curr_pts = best_pts.copy()
    for f in range(best_k - 1, -1, -1):
        g_curr = cv2.cvtColor(buffered_frames[f + 1], cv2.COLOR_BGR2GRAY)
        g_prev = cv2.cvtColor(buffered_frames[f], cv2.COLOR_BGR2GRAY)
        p_feat = cv2.goodFeaturesToTrack(g_curr, maxCorners=250, qualityLevel=0.03, minDistance=15, mask=court_mask)
        if p_feat is not None and len(p_feat) >= 10:
            p_next, status, _ = cv2.calcOpticalFlowPyrLK(g_curr, g_prev, p_feat, None)
            valid = (status.reshape(-1) == 1)
            if valid.sum() >= 8:
                H, _ = cv2.findHomography(p_feat.reshape(-1, 2)[valid], p_next.reshape(-1, 2)[valid], cv2.RANSAC)
                if H is not None:
                    curr_pts = cv2.perspectiveTransform(curr_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        kps_buffer[f] = curr_pts.copy()

    # Forward tracking from best_k to end of buffer
    curr_pts = best_pts.copy()
    for f in range(best_k + 1, len(buffered_frames)):
        g_prev = cv2.cvtColor(buffered_frames[f - 1], cv2.COLOR_BGR2GRAY)
        g_curr = cv2.cvtColor(buffered_frames[f], cv2.COLOR_BGR2GRAY)
        p_feat = cv2.goodFeaturesToTrack(g_prev, maxCorners=250, qualityLevel=0.03, minDistance=15, mask=court_mask)
        if p_feat is not None and len(p_feat) >= 10:
            p_next, status, _ = cv2.calcOpticalFlowPyrLK(g_prev, g_curr, p_feat, None)
            valid = (status.reshape(-1) == 1)
            if valid.sum() >= 8:
                H, _ = cv2.findHomography(p_feat.reshape(-1, 2)[valid], p_next.reshape(-1, 2)[valid], cv2.RANSAC)
                if H is not None:
                    curr_pts = cv2.perspectiveTransform(curr_pts.reshape(-1, 1, 2), H).reshape(-1, 2)
        kps_buffer[f] = curr_pts.copy()

    # Populate all_kps for buffered frames
    all_kps = [kps_buffer[i].reshape(-1).copy() for i in range(len(buffered_frames))]

    # Setup state for continuing forward tracking beyond buffer
    last_frame = buffered_frames[-1]
    prev_gray = cv2.cvtColor(last_frame, cv2.COLOR_BGR2GRAY)
    current_pts = kps_buffer[-1].copy()
    anchor_pts = current_pts.copy()
    prev_pts_features = cv2.goodFeaturesToTrack(
        prev_gray,
        maxCorners=250,
        qualityLevel=0.03,
        minDistance=15,
        mask=court_mask
    )
    anchor_pts_features = prev_pts_features.copy() if prev_pts_features is not None else None
    lost_frames = 0
    last_redetection = len(buffered_frames) - 1
    last_accepted_detection = len(buffered_frames) - 1

    # 4. Tracking liên tục qua từng frame còn lại
    for f_idx, frame in enumerate(frames, start=len(buffered_frames)):
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        transform_applied = False
        camera_motion = 0.0
        static_scene = False
        refresh_features = False

        if prev_pts_features is not None and anchor_pts_features is not None and len(prev_pts_features) >= 10:
            next_pts_features, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_pts_features, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )
            if next_pts_features is None or status is None:
                good_prev = np.empty((0, 2), dtype=np.float32)
                good_next = np.empty((0, 2), dtype=np.float32)
                good_anchor = np.empty((0, 2), dtype=np.float32)
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
                good_anchor = anchor_pts_features.reshape((-1, 2))[valid]

            if len(good_prev) >= 8:
                accumulated_flow = np.linalg.norm(good_next - good_anchor, axis=1)
                static_scene = (
                    np.median(accumulated_flow) < 0.6
                    and np.percentile(accumulated_flow, 90) < 1.2
                )
                if static_scene:
                    # A brief low-flow estimate must not snap points back to an old anchor.
                    if np.median(np.linalg.norm(current_pts - anchor_pts, axis=1)) < 0.6:
                        current_pts = anchor_pts.copy()
                    else:
                        refresh_features = True
                    transform_applied = True
                else:
                    matrix, inlier_ratio = estimate_court_homography(good_anchor, good_next)
                    if matrix is not None and inlier_ratio >= 0.5:
                        transformed_points = cv2.perspectiveTransform(
                            anchor_pts.reshape((-1, 1, 2)), matrix
                        ).reshape((-1, 2))
                        point_step = np.linalg.norm(transformed_points - current_pts, axis=1)
                        feature_step = np.linalg.norm(good_next - good_prev, axis=1)
                        step_limit = max(8.0, 6.0 * np.percentile(feature_step, 90))
                        if np.isfinite(transformed_points).all() and np.percentile(point_step, 90) <= step_limit:
                            camera_motion = float(np.median(point_step))
                            current_pts = transformed_points
                            transform_applied = True
                        else:
                            # Bad feature geometry: keep the last good court and start a fresh keyframe.
                            refresh_features = True

            if len(good_next) < 80:
                refresh_features = True
            else:
                prev_pts_features = good_next.reshape(-1, 1, 2)
                anchor_pts_features = good_anchor.reshape(-1, 1, 2)
        else:
            refresh_features = True

        lost_frames = 0 if transform_applied else lost_frames + 1
        periodic_redetection = redetect_interval and f_idx % redetect_interval == 0
        recovery_redetection = lost_frames >= 12
        motion_redetection = camera_motion >= 2.5
        if detector is not None and f_idx - last_redetection >= 12 and (
            periodic_redetection or recovery_redetection or motion_redetection
        ):
            last_redetection = f_idx
            try:
                detected_points = np.asarray(detector.predict(frame), dtype=np.float32).reshape((-1, 2))
                if detected_points.shape == (14, 2) and np.isfinite(detected_points).all():
                    disagreement = np.median(np.linalg.norm(detected_points - current_pts, axis=1))
                    tracked_score = court_line_contrast(curr_gray, current_pts)
                    detected_score = court_line_contrast(curr_gray, detected_points)
                    supported = detected_score >= 15.0 and (
                        (disagreement <= 3.0 and detected_score >= tracked_score - 5.0)
                        or (disagreement <= 20.0 and detected_score >= tracked_score + 15.0)
                    )
                    if supported:
                        if disagreement > 1.0 and not recovery_redetection:
                            start = max(last_accepted_detection, f_idx - 60)
                            correction = (detected_points - current_pts).reshape(-1)
                            for index in range(start + 1, f_idx):
                                if tracked_score >= 15.0:
                                    fraction = (index - start) / float(f_idx - start)
                                    all_kps[index] += correction * fraction
                                else:
                                    all_kps[index] += correction
                        current_pts = detected_points
                        lost_frames = 0
                        last_accepted_detection = f_idx
                        refresh_features = True
            except RuntimeError:
                pass

        if refresh_features:
            anchor_pts = current_pts.copy()
            prev_pts_features = cv2.goodFeaturesToTrack(
                curr_gray, maxCorners=250, qualityLevel=0.03, minDistance=15, mask=court_mask
            )
            anchor_pts_features = prev_pts_features.copy() if prev_pts_features is not None else None

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
