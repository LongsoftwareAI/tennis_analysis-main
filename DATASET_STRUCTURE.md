# 🎾 CẤU TRÚC VÀ HƯỚNG DẪN QUẢN LÝ DỮ LIỆU (DATASET STRUCTURE)

Tài liệu này hướng dẫn chi tiết cách tổ chức dữ liệu, phân loại giữa **Dữ liệu phát hiện bóng (Ball Detection)** và **Dữ liệu điểm mốc vạch sân (Court Keypoints)**, cùng các lệnh huấn luyện tương ứng.

---

## 📁 1. Cây Thư Mục Tổng Quan

```text
tennis_analysis-main/
│
├── training/
│   ├── datasets/
│   │   ├── ball/                                # 🔴 DỮ LIỆU BÓNG TENNIS (TENNIS BALL)
│   │   │   ├── dataset_1_tennis_ball/           # Dataset 1 (Viren Dhanwani) - 577 ảnh bóng
│   │   │   ├── dataset_2_ball_detection/         # Dataset 2 (Tennis Project) - 2,370 ảnh bóng
│   │   │   ├── dataset_3_me_tennis/              # Dataset 3 (me-dxtf3) - 1,473 ảnh bóng
│   │   │   └── merged_tennis_dataset/           # ⭐ BỘ GỘP CHUẨN ĐỂ HUẤN LUYỆN (4,454 ảnh - 4,420 nhãn bóng)
│   │   │       ├── data.yaml                    # File cấu hình (nc: 1, name: tennis_ball)
│   │   │       ├── train/ (images, labels)      # 3,731 ảnh huấn luyện
│   │   │       ├── valid/ (images, labels)      # 426 ảnh kiểm định
│   │   │       └── test/  (images, labels)      # 297 ảnh kiểm thử
│   │   │
│   │   └── court/                               # 🟢 DỮ LIỆU ĐIỂM MỐC VẠCH SÂN (COURT KEYPOINTS)
│   │       └── tennis_court_keypoints/          # Dataset 10,000 ảnh từ Roboflow (me-dxtf3)
│   │           ├── data.json                    # File nhãn tọa độ 14 keypoints (28 giá trị) cho TensorFlow
│   │           ├── data.yaml                    # Cấu hình 14 classes (1..14)
│   │           ├── train/ (images, labels)      # 7,909 ảnh
│   │           ├── valid/ (images, labels)      # 1,579 ảnh
│   │           └── test/  (images, labels)      # 512 ảnh
│   │
│   ├── tennis_ball_detector_training_yolo26.ipynb   # Notebook huấn luyện bóng (GPU RTX 5060 + Early Stopping)
│   ├── tennis_court_keypoints_training_tf.ipynb     # Notebook huấn luyện vạch sân (100% TensorFlow ResNet50)
│   ├── train_yolo26_ball_detector.py                # Script chạy dòng lệnh huấn luyện bóng
│   ├── train_court_line_detector_tf.py              # Script chạy dòng lệnh huấn luyện vạch sân
│   └── check_merged_dataset.py                      # Script kiểm tra tính toàn vẹn bộ bóng gộp
│
├── models/                                      # Chứa các checkpoint mô hình đã huấn luyện
├── reports/                                     # Biểu đồ đánh giá huấn luyện (Loss, mAP, Confusion Matrix)
├── trackers/                                    # Module theo dõi người và bóng
└── main.py                                      # Ứng dụng chính phân tích trận đấu tennis
```

---

## 🎯 2. Phân Biệt Hai Nhóm Dữ Liệu

### A. Nhóm Dữ Liệu Phát Hiện Quả Bóng (`training/datasets/ball/`)
- **Mục đích:** Huấn luyện mô hình YOLO26 Large phát hiện quả bóng tennis nhỏ, bay nhanh ở mọi điều kiện ánh sáng.
- **Định dạng nhãn:** YOLO Detection Format (`.txt`):
  `0 <x_center> <y_center> <width> <height>` (tất cả tọa độ được chuẩn hóa `[0, 1]`).
- **Thư mục sử dụng:** `merged_tennis_dataset/` là bộ dữ liệu gộp từ cả 3 nguồn (4,454 ảnh, 4,420 nhãn bóng, 1 class duy nhất: `tennis_ball`).
- **Cách huấn luyện:**
  - **Cách 1 (Jupyter Notebook - Khuyên dùng):** Mở [tennis_ball_detector_training_yolo26.ipynb](file:///d:/FPT/DAT301m/tennis_analysis-main/training/tennis_ball_detector_training_yolo26.ipynb), chọn Kernel `Python (tennis_venv)`, bấm **Run All**.
  - **Cách 2 (Terminal):**
    ```powershell
    .\venv\Scripts\python.exe training/train_yolo26_ball_detector.py --dataset merged --epochs 50 --batch 8 --patience 10
    ```

---

### B. Nhóm Dữ Liệu Điểm Mốc Vạch Sân (`training/datasets/court/`)
- **Mục đích:** Huấn luyện mạng nơ-ron tích chập ResNet50 trong **TensorFlow / Keras** xác định 14 điểm mốc then chốt trên sân để tính toán biến đổi phối cảnh 2D Mini-court (xác định bóng In/Out và tốc độ di chuyển cầu thủ).
- **Định dạng nhãn:** File [`data.json`](file:///d:/FPT/DAT301m/tennis_analysis-main/training/datasets/court/tennis_court_keypoints/data.json) chứa 14 cặp tọa độ $(x_1, y_1, x_2, y_2, \dots, x_{14}, y_{14})$ (28 số thực) cho từng ảnh.
- **Quy mô:** **10,000 ảnh** sân tennis thực tế.
- **Cách huấn luyện:**
  - **Cách 1 (Jupyter Notebook):** Mở [tennis_court_keypoints_training_tf.ipynb](file:///d:/FPT/DAT301m/tennis_analysis-main/training/tennis_court_keypoints_training_tf.ipynb), chọn Kernel `Python (tennis_venv)`, bấm **Run All**.
  - **Cách 2 (Terminal):**
    ```powershell
    .\venv\Scripts\python.exe training/train_court_line_detector_tf.py --epochs 50 --batch_size 16
    ```

---

## 🛠️ 3. Các Lệnh Tiện Ích Hỗ Trợ

1. **Kiểm tra tính toàn vẹn của bộ dữ liệu bóng gộp:**
   ```powershell
   .\venv\Scripts\python.exe training/check_merged_dataset.py
   ```
2. **Chạy phân tích video hoàn chỉnh sau khi có model:**
   ```powershell
   .\venv\Scripts\python.exe main.py
   ```
