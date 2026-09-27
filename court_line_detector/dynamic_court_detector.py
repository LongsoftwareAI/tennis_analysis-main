import os
import pickle
import numpy as np
import pandas as pd

def detect_court_keypoints_dynamically(
    court_line_detector,
    video_frames,
    sample_interval=2,
    smooth_window=5,
    stub_path=None
):
    """
    [CÁCH 2 - DYNAMIC PER-FRAME COURT DETECTION]
    Hàm phát hiện toạ độ sân động theo từng frame (hoặc theo chu kỳ sample_interval).
    Tự động thích ứng khi máy quay lia (pan), nghiêng (tilt) hoặc phóng to/thu nhỏ (zoom).

    Được thiết kế độc lập (modular):
    - Dễ dàng gắn vào pipeline bằng 1 dòng gọi hàm.
    - Dễ dàng gỡ bỏ hoặc quay lại chế độ tĩnh ban đầu chỉ bằng cách uncomment dòng cũ.

    Parameters:
    -----------
    court_line_detector : CourtLineDetector
        Đối tượng detector đã được khởi tạo.
    video_frames : list of np.ndarray
        Danh sách tất cả các frames của video.
    sample_interval : int, default=2
        Chu kỳ quét (1 = quét từng frame; 2 = quét cách 1 frame rồi nội suy tuyến tính).
        sample_interval=2 giúp tăng tốc độ xử lý gấp đôi mà vẫn giữ độ mượt mà tuyệt đối.
    smooth_window : int, default=5
        Độ rộng cửa sổ lọc mượt thời gian (Rolling Mean Filter) để loại bỏ rung giật pixel.
    stub_path : str, optional
        Đường dẫn file cache (.pkl). Nếu đã có cache, sẽ load ngay trong 0.01s.

    Returns:
    --------
    np.ndarray of shape (len(video_frames), 28)
        Mảng 2D chứa 14 toạ độ (x, y) riêng biệt cho từng frame tương ứng.
    """
    total_frames = len(video_frames)
    if total_frames == 0:
        return np.array([])

    # 1. Kiểm tra cache nếu có stub_path
    if stub_path and os.path.exists(stub_path):
        print(f"[DynamicCourt] Loading cached dynamic keypoints from {stub_path}")
        with open(stub_path, "rb") as f:
            return pickle.load(f)

    print(f"[DynamicCourt] Starting dynamic court detection across {total_frames} frames (interval={sample_interval})...")

    # 2. Quét keypoints theo chu kỳ sample_interval
    raw_keypoints = np.zeros((total_frames, 28), dtype=np.float32)
    sampled_indices = list(range(0, total_frames, sample_interval))
    if sampled_indices[-1] != total_frames - 1:
        sampled_indices.append(total_frames - 1)

    for idx, f_idx in enumerate(sampled_indices):
        kps = court_line_detector.predict(video_frames[f_idx])
        raw_keypoints[f_idx] = kps
        if (idx + 1) % 25 == 0 or (idx + 1) == len(sampled_indices):
            print(f"[DynamicCourt] Processed {idx + 1}/{len(sampled_indices)} keyframe predictions...")

    # 3. Nội suy tuyến tính (Linear Interpolation) cho các frame nằm giữa
    df = pd.DataFrame(raw_keypoints)
    # Gán các frame chưa quét bằng NaN để interpolate
    non_sampled_mask = np.ones(total_frames, dtype=bool)
    non_sampled_mask[sampled_indices] = False
    df.loc[non_sampled_mask] = np.nan

    df_interpolated = df.interpolate(method='linear').bfill().ffill()

    # 4. Lọc khử nhiễu ngoại lai (Outlier Rejection) & Làm mượt thời gian (Temporal Smoothing)
    # Dùng Rolling Median để loại bỏ bất kỳ cú nhảy điểm bất thường nào do cầu thủ/vợt che vạch
    median_df = df_interpolated.rolling(window=7, min_periods=1, center=True).median()
    deviation = (df_interpolated - median_df).abs()
    # Nếu lệch quá 12px so với median xung quanh, thay bằng median
    df_cleaned = df_interpolated.where(deviation < 12.0, median_df)

    # Làm mượt nhẹ bằng Rolling Mean với center=True
    if smooth_window > 1:
        smoothed_df = df_cleaned.rolling(window=smooth_window, min_periods=1, center=True).mean()
    else:
        smoothed_df = df_cleaned

    final_keypoints = smoothed_df.to_numpy().astype(np.float32)

    # 5. Lưu cache stub nếu có chỉ định
    if stub_path:
        os.makedirs(os.path.dirname(stub_path), exist_ok=True)
        with open(stub_path, "wb") as f:
            pickle.dump(final_keypoints, f)
        print(f"[DynamicCourt] Cached dynamic keypoints to {stub_path}")

    return final_keypoints
