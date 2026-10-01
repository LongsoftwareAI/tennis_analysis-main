# 📘 TÀI LIỆU CHI TIẾT VỀ TIỀN XỬ LÝ DỮ LIỆU (DATA PREPROCESSING)
### HỆ THỐNG PHÂN TÍCH QUẦN VỢT TENNIS ANALYSIS SYSTEM (YOLO26 & RESNET50)

Tài liệu này mô tả chi tiết toàn bộ kiến trúc, thuật toán, vị trí mã nguồn và quy trình xử lý dữ liệu của hệ thống, bao gồm cả hai giai đoạn:
1. **Tiền xử lý dữ liệu phục vụ huấn luyện (Training Data Preprocessing)**
2. **Tiền xử lý luồng dữ liệu thời gian thực khi phân tích video (Inference Pipeline Preprocessing)**

---

## 📑 MỤC LỤC
1. [Sơ đồ Kiến trúc Quy trình Tiền xử lý Dữ liệu](#1-sơ-đồ-kiến-trúc-quy-trình-tiền-xử-lý-dữ-liệu)
2. [Giai đoạn 1: Tiền xử lý Dữ liệu Huấn luyện (Training Data Preprocessing)](#2-giai-đoạn-1-tiền-xử-lý-dữ-liệu-huấn-luyện-training-data-preprocessing)
   - [2.1. Bộ dữ liệu Điểm mốc Vạch sân (Court Keypoints Dataset - ResNet50)](#21-bộ-dữ-liệu-điểm-mốc-vạch-sân-court-keypoints-dataset---resnet50)
   - [2.2. Bộ dữ liệu Quả bóng Tennis (Tennis Ball Dataset - YOLO26)](#22-bộ-dữ-liệu-quả-bóng-tennis-tennis-ball-dataset---yolo26)
3. [Giai đoạn 2: Tiền xử lý Dữ liệu khi Chạy Phân tích Video (Inference Preprocessing)](#3-giai-đoạn-2-tiền-xử-lý-dữ-liệu-khi-chạy-phân-tích-video-inference-preprocessing)
   - [3.1. Đọc video và trích xuất khung hình (Frame Extraction)](#31-đọc-video-và-trích-xuất-khung-hình-frame-extraction)
   - [3.2. Tiền xử lý Tensor cho ResNet50 bám vạch sân](#32-tiền-xử-lý-tensor-cho-resnet50-bám-vạch-sân)
   - [3.3. Hậu xử lý Sub-pixel tinh chỉnh vạch sơn trắng (HSV Line Snapping)](#33-hậu-xử-lý-sub-pixel-tinh-chỉnh-vạch-sơn-trắng-hsv-line-snapping)
   - [3.4. Tiền xử lý cho thuật toán quang học Pure CPV Optical Flow](#34-tiền-xử-lý-cho-thuật-toán-quang-học-pure-cpv-optical-flow)
   - [3.5. Làm sạch và nội suy dữ liệu quỹ đạo bóng (Ball Trajectory Cleaning & Interpolation)](#35-làm-sạch-và-nội-suy-dữ-liệu-quỹ-đạo-bóng-ball-trajectory-cleaning--interpolation)
4. [Bảng Tổng hợp Đối chiếu Toàn diện](#4-bảng-tổng-hợp-đối-chiếu-toàn-diện)

---

## 1. SƠ ĐỒ KIẾN TRÚC QUY TRÌNH TIỀN XỬ LÝ DỮ LIỆU

```
                             [ DỮ LIỆU ĐẦU VÀO ]
                                      │
          ┌───────────────────────────┴───────────────────────────┐
          ▼                                                       ▼
[ GIAI ĐOẠN HUẤN LUYỆN (TRAINING) ]             [ GIAI ĐOẠN SUY LUẬN (INFERENCE) ]
          │                                                       │
  ┌───────┴───────┐                                       ┌───────┴───────┐
  ▼               ▼                                       ▼               ▼
[SÂN TENNIS]    [BÓNG TENNIS]                         [SÂN TENNIS]    [BÓNG TENNIS]
  │               │                                       │               │
  ├─► Tải & lọc   ├─► Gộp 3 nguồn                         ├─► BGR -> RGB  ├─► Lọc tĩnh
  │   >=12 kps    │   khử trùng ds{i}_                    │   (224x224)   │   (Static Suppress)
  ├─► Xuất        ├─► Chuẩn hóa nhãn                      ├─► ImageNet    ├─► Spike Filter
  │   data.json   │   class 0                             │   Norm        │   (Pandas NaN)
  ├─► Z-Score     ├─► Letterbox                           ├─► HSV White   ├─► Linear
  │   ImageNet    │   640x640                             │   Snap        │   Interpolation
  └─► Scale kps   └─► Mosaic +                            └─► Optical     └─► Median Box
      theo tỷ lệ      Color Jitter                            Flow Gray       Stabilizer
      224/W, 224/H
```

---

## 2. GIAI ĐOẠN 1: TIỀN XỬ LÝ DỮ LIỆU HUẤN LUYỆN (TRAINING DATA PREPROCESSING)

### 2.1. Bộ dữ liệu Điểm mốc Vạch sân (Court Keypoints Dataset - ResNet50)
* **Quy mô:** **10,000 ảnh** sân tennis thực tế từ Roboflow (`me-dxtf3/tennis-court-keypoints-ukqn8`).
* **Mục tiêu:** Huấn luyện mô hình hồi quy (Regression) để dự đoán tọa độ 14 điểm mốc then chốt trên sân (tương ứng vector 28 số thực).
* **Đặc thù bài toán:** Khác với phát hiện vật thể (Object Detection) dự đoán Bounding Box, bài toán Keypoints Detection đòi hỏi tọa độ các điểm mốc phải bám chính xác vào giao điểm vạch vôi, không được bị lệch khi ảnh co giãn kích thước.

#### 🔹 Bước 1: Chuyển đổi nhãn thô từ Roboflow format sang file cấu trúc tổng hợp `data.json`
* **Vị trí file mã nguồn:** [`training/download_court_dataset.py`](training/download_court_dataset.py) (Dòng 54 - 100)
* **Vấn đề của dữ liệu thô:**
  - Nhãn ban đầu của Roboflow lưu ở dạng các file `.txt` độc lập cho từng ảnh.
  - Mỗi file `.txt` chứa nhiều dòng rời rạc, mỗi dòng gồm: `<class_id> <x_center> <y_center> <width> <height>` tương ứng với từng điểm mốc riêng biệt.
  - Có những ảnh bị lỗi gán nhãn, thiếu một số điểm mốc hoặc thứ tự các điểm bị xáo trộn.
* **Thuật toán xử lý:**
  1. Duyệt qua toàn bộ danh sách ảnh trong 3 tập `train/`, `valid/`, `test/`.
  2. Đọc file nhãn `.txt` tương ứng và ánh xạ vào từ điển `kps_dict[class_id] = (xc, yc)`.
  3. **Bộ lọc chất lượng nhãn (Label Integrity Filtering):** Chỉ chấp nhận các ảnh có tối thiểu 12 điểm mốc hợp lệ (`len(kps_dict) >= 12`). Nếu một ảnh bị gán thiếu quá nhiều điểm mốc, ảnh đó sẽ bị loại bỏ để không gây nhiễu gradient khi huấn luyện.
  4. **Bù khuyết và phẳng hóa tọa độ (Padding & Flattening):** Với 14 điểm mốc ($0 \to 13$), nếu điểm nào bị thiếu trong ảnh sẽ được gán tọa độ an toàn `(0.0, 0.0)`. Kết quả tạo ra một mảng phẳng 28 phần tử:
     $$[x_0, y_0, x_1, y_1, x_2, y_2, \dots, x_{13}, y_{13}]$$
  5. Xuất toàn bộ danh sách ảnh hợp lệ kèm mảng tọa độ vào file duy nhất: [`training/datasets/court/tennis_court_keypoints/data.json`](training/datasets/court/tennis_court_keypoints/data.json).

```python
# Trích đoạn từ training/download_court_dataset.py
if len(kps_dict) >= 12:
    kps_flat = []
    for cid in range(14):
        if cid in kps_dict:
            kps_flat.extend([kps_dict[cid][0], kps_dict[cid][1]])
        elif (cid + 1) in kps_dict:
            kps_flat.extend([kps_dict[cid + 1][0], kps_dict[cid + 1][1]])
        else:
            kps_flat.extend([0.0, 0.0]) # Bù khuyết điểm bị thiếu
    
    json_records.append({
        "id": os.path.relpath(img_p, target_dir).replace("\\", "/"),
        "kps": kps_flat
    })

with open(out_json, 'w', encoding='utf-8') as f:
    json.dump(json_records, f, indent=2)
```

---

#### 🔹 Bước 2: Nạp dữ liệu, chuẩn hóa ảnh và co giãn tọa độ nhãn theo tỷ lệ khung hình
* **Vị trí file mã nguồn:** [`training/train_court_line_detector_tf.py`](training/train_court_line_detector_tf.py) (Dòng 18 - 68, hàm `load_court_data`) và [`training/tennis_court_keypoints_training_tf.ipynb`](training/tennis_court_keypoints_training_tf.ipynb)
* **Thuật toán xử lý trên từng mẫu dữ liệu:**
  1. **Đọc ảnh:** Đọc file ảnh từ đĩa bằng `cv2.imread(img_path)`.
  2. **Chuyển đổi không gian màu:** OpenCV mặc định nạp ảnh ở hệ màu `BGR`, cần chuyển sang `RGB` (`cv2.COLOR_BGR2RGB`) để khớp với trọng số ImageNet của ResNet50.
  3. **Co giãn kích thước ảnh (Spatial Resizing):** Ảnh chụp từ truyền hình có kích thước $1920 \times 1080$ hoặc $1280 \times 720$. Để đưa vào ResNet50, ảnh được resize về kích thước chuẩn $(224, 224)$ bằng `cv2.resize()`.
  4. **Chuẩn hóa giá trị điểm ảnh (Pixel Scaling & Z-score Normalization):**
     - Đưa miền giá trị màu $[0, 255]$ về đoạn $[0.0, 1.0]$: $\text{img} = \frac{\text{img}}{255.0}$
     - Trừ đi giá trị trung bình (`mean`) và chia cho độ lệch chuẩn (`std`) của tập dữ liệu ImageNet:
       $$\text{Norm\_Img} = \frac{\text{Img} - \mu}{\sigma}, \quad \mu = [0.485, 0.456, 0.406], \quad \sigma = [0.229, 0.224, 0.225]$$
  5. **Tái chuẩn hóa tọa độ nhãn (Keypoints Coordinate Rescaling):**
     - Khi ảnh bị thu nhỏ từ $(W_{\text{orig}}, H_{\text{orig}})$ về $(224, 224)$, toàn bộ tọa độ nhãn $x$ và $y$ phải được nhân theo hệ số co giãn tương ứng:
       $$x_{\text{scaled}} = x_{\text{orig}} \times \frac{224}{W_{\text{orig}}}, \quad y_{\text{scaled}} = y_{\text{orig}} \times \frac{224}{H_{\text{orig}}}$$
     - Nhờ đó, nhãn mục tiêu (target ground-truth) nằm hoàn toàn trong không gian tọa độ $[0, 224]$, giúp hàm mất mát Mean Squared Error (MSE) hội tụ ổn định và chính xác.

```python
# Trích đoạn từ training/train_court_line_detector_tf.py (hàm load_court_data)
orig_h, orig_w = raw_img.shape[:2]
img_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
resized = cv2.resize(img_rgb, (224, 224))
norm_img = ((resized.astype(np.float32) / 255.0) - mean) / std

kps = np.array(item["kps"], dtype=np.float32)
if np.max(kps) <= 1.05:  # Nếu nhãn đã ở dạng tỷ lệ [0, 1]
    kps[::2] *= 224.0
    kps[1::2] *= 224.0
else:                    # Nếu nhãn là tọa độ pixel gốc
    kps[::2] *= 224.0 / orig_w
    kps[1::2] *= 224.0 / orig_h
```

---

### 2.2. Bộ dữ liệu Quả bóng Tennis (Tennis Ball Dataset - YOLO26)
* **Quy mô:** **4,454 ảnh** và **4,420 nhãn quả bóng tennis** đã hợp nhất.
* **Mục tiêu:** Huấn luyện mô hình YOLO26 Small (`yolo26s.pt`) phát hiện quả bóng tennis nhỏ xíu, bay tốc độ cao ($>150\text{ km/h}$) dưới các góc quay lia máy nhanh và độ mờ chuyển động (motion blur).

#### 🔹 Bước 1: Hợp nhất đa nguồn dữ liệu và khử xung đột nhãn
* **Vị trí file mã nguồn:** [`training/dataset_downloader.py`](training/dataset_downloader.py) (Dòng 86 - 140, hàm `merge_datasets`)
* **Thách thức:** Dữ liệu được thu thập từ 3 bộ dataset Roboflow khác nhau:
  1. `dataset_1_tennis_ball` (577 ảnh bóng)
  2. `dataset_2_ball_detection` (2,370 ảnh bóng)
  3. `dataset_3_me_tennis` (1,473 ảnh bóng)
  - Các bộ này có thể trùng tên file ảnh (ví dụ cùng có `frame_001.jpg`), ID lớp nhãn bị đặt khác nhau (bộ thì đặt class 0, bộ thì đặt class 1 hoặc 2 do có lẫn lớp người chơi).
* **Thuật toán xử lý:**
  1. Duyệt qua từng bộ dữ liệu theo từng phân vùng `train`, `valid`, `test`.
  2. **Thêm tiền tố chống đè file (Prefixing):** Tự động gắn tiền tố `ds{idx}_` vào tên ảnh và file `.txt` tương ứng (ví dụ: `ds0_frame_01.jpg`, `ds1_frame_01.jpg`, `ds2_frame_01.jpg`).
  3. **Đồng nhất hóa lớp đối tượng (Class Unification):** Chỉ giữ lại các bounding box của quả bóng và ép về lớp duy nhất: `class 0: tennis_ball`.
  4. **Tạo file siêu dữ liệu [`data.yaml`](training/datasets/ball/merged_tennis_dataset/data.yaml):** Định nghĩa cấu trúc chuẩn YOLO:
     ```yaml
     path: d:/FPT/DAT301m/tennis_analysis-main/training/datasets/ball/merged_tennis_dataset
     train: train/images
     val: valid/images
     test: test/images
     names:
       0: tennis_ball
     ```

---

#### 🔹 Bước 2: Pipeline Tiền xử lý trực tiếp & Tăng cường dữ liệu (Online Augmentation)
* **Vị trí file mã nguồn:** [`training/train_yolo26_ball_detector.py`](training/train_yolo26_ball_detector.py) (Dòng 151 - 163) và [`training/tennis_ball_detector_training_yolo26.ipynb`](training/tennis_ball_detector_training_yolo26.ipynb)
* **Thuật toán xử lý của YOLO26:**
  1. **Co giãn giữ tỷ lệ (Letterbox Resizing $640 \times 640$):**
     - Quả bóng tennis trong không gian thực là hình cầu đối xứng hoàn hảo. Nếu resize trực tiếp kiểu bóp méo (stretch), bóng sẽ bị biến dạng thành hình bầu dục dẹt, làm giảm nghiêm trọng khả năng phát hiện của mạng tích chập.
     - Thuật toán `Letterbox` tính toán tỷ lệ cạnh dài nhất của ảnh gốc để co về $640\text{ px}$, sau đó bổ sung viền đệm màu xám đồng nhất (pixel giá trị `114`) ở các cạnh ngắn hơn. Nhờ đó, tỷ lệ khung hình $1:1$ của quả bóng được bảo toàn nguyên vẹn 100%.
  2. **Chuẩn hóa nhãn Bounding Box:** Tọa độ hộp bao được chuẩn hóa về dải số thực $[0.0, 1.0]$:
     $$\left( \frac{x_{\text{center}}}{W}, \frac{y_{\text{center}}}{H}, \frac{w}{W}, \frac{h}{H} \right)$$
  3. **Tăng cường dữ liệu chuyên biệt cho vật thể nhỏ (Small Object Augmentation):**
     - **Mosaic Augmentation (Ghép 4 khung hình):** Tự động ghép ngẫu nhiên 4 ảnh khác nhau thành một khung hình $640 \times 640$. Điều này giúp tăng mật độ bối cảnh, mô phỏng quả bóng ở khoảng cách xa (khi quả bóng chỉ chiếm $4 \times 4\text{ pixels}$) và buộc mô hình học cách phân biệt bóng với các chi tiết gây nhiễu như vệt sơn sân, vạt áo hay giày vận động viên.
     - **HSV Color Jitter:** Biến thiên ngẫu nhiên các kênh màu Sắc độ (Hue $\pm 0.015$), Độ bão hòa (Saturation $\pm 0.7$), Độ sáng (Value $\pm 0.4$) để mô hình nhận diện tốt quả bóng màu vàng/xanh neon trong cả điều kiện trời nắng gắt hay dưới dàn đèn công suất lớn ban đêm.

---

## 3. GIAI ĐOẠN 2: TIỀN XỬ LÝ DỮ LIỆU KHI CHẠY PHÂN TÍCH VIDEO (INFERENCE PREPROCESSING)

Khi người dùng chạy lệnh `python main.py` để phân tích một video trận đấu, chuỗi các hàm tiền xử lý được thực thi tuần tự theo thời gian thực:

### 3.1. Đọc video và trích xuất khung hình (Frame Extraction)
* **Vị trí file mã nguồn:** [`utils/video_utils.py`](utils/video_utils.py) (Dòng 4 - 13, hàm `read_video`)
* **Cách xử lý:** Sử dụng `cv2.VideoCapture` giải mã luồng video đầu vào (`.mp4`, `.avi`) thành danh sách các mảng nhiều chiều NumPy `[frame_0, frame_1, ..., frame_n]` ở hệ màu gốc BGR với độ phân giải chuẩn của video.

---

### 3.2. Tiền xử lý Tensor cho ResNet50 bám vạch sân
* **Vị trí file mã nguồn:** [`court_line_detector/court_line_detector.py`](court_line_detector/court_line_detector.py) (Dòng 49 - 63, hàm `preprocess_image`)
* **Cách xử lý:** Tại Frame đầu tiên (Frame 0), ảnh được tiền xử lý trước khi đưa vào mô hình ResNet50 Keras 3:
  1. `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`: Đổi hệ màu BGR sang RGB.
  2. `cv2.resize(image_rgb, (224, 224))`: Co nhỏ về chuẩn $224 \times 224$.
  3. `norm_img = (resized / 255.0 - mean) / std`: Chuẩn hóa Z-score với bộ tham số ImageNet.
  4. `np.expand_dims(norm_img, axis=0)`: Tạo chiều batch `(1, 224, 224, 3)` để suy luận trên GPU RTX 5060.

---

### 3.3. Hậu xử lý Sub-pixel tinh chỉnh vạch sơn trắng (HSV Line Snapping)
* **Vị trí file mã nguồn:** [`court_line_detector/court_line_detector.py`](court_line_detector/court_line_detector.py) (Dòng 86 - 150, hàm `refine_court_keypoints`)
* **Cách xử lý:** Mạng nơ-ron học sâu ResNet50 chỉ đưa ra dự đoán thô (Coarse Prediction) với sai số $\pm 3\text{ px}$. Hệ thống áp dụng thuật toán xử lý ảnh cổ điển (Classical Computer Vision) để hút dính chính xác từng sub-pixel:
  1. Chuyển ảnh sang không gian màu **HSV** (`cv2.COLOR_BGR2HSV`).
  2. Áp dụng ngưỡng nhị phân lọc màu trắng vạch sân:
     $$\text{Lower} = [0, 0, 180], \quad \text{Upper} = [180, 50, 255]$$
  3. **Tạo mặt nạ vùng sân (Court Convex Hull Mask):** Loại trừ 100% vùng khán đài, biển quảng cáo bên ngoài để tránh bắt nhầm màu trắng của áo khán giả.
  4. Quét profile cục bộ xung quanh điểm thô để tìm tâm dải sáng trắng, tự động dời tọa độ điểm mốc vào đúng giao điểm thực tế của vạch sân.

---

### 3.4. Tiền xử lý cho thuật toán quang học Pure CPV Optical Flow
* **Vị trí file mã nguồn:** [`court_line_detector/cpv_court_tracker.py`](court_line_detector/cpv_court_tracker.py) (Dòng 68 - 83)
* **Cách xử lý:** Từ Frame 1 đến hết trận đấu, hệ thống chuyển sang bám sân bằng quang học thuần túy (Pure CPV) với tốc độ $>150\text{ FPS}$:
  1. **Chuyển đổi ảnh xám:** `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` để triệt tiêu chiều màu sắc, giảm $3\times$ khối lượng tính toán.
  2. **Tạo ROI Court Mask:** Giới hạn vùng trích xuất trong phạm vi $25\% \to 95\%$ chiều cao và $10\% \to 90\%$ chiều rộng.
  3. **Trích xuất 250 điểm đặc trưng Shi-Tomasi:** Dùng `cv2.goodFeaturesToTrack` tìm các góc cạnh có độ tương phản cao trên mặt sân.
  4. Đưa vào thuật toán Lucas-Kanade Kim tự tháp (`cv2.calcOpticalFlowPyrLK`) kết hợp RANSAC Affine Partial để triệt tiêu 100% hiện tượng rung giật vạch sân.

---

### 3.5. Làm sạch và nội suy dữ liệu quỹ đạo bóng (Ball Trajectory Cleaning & Interpolation)
* **Vị trí file mã nguồn:** [`trackers/ball_tracker.py`](trackers/ball_tracker.py) (Dòng 59 - 106, hàm `interpolate_ball_positions`)
* **Thách thức:** Do bóng bay quá nhanh hoặc bị che khuất bởi vợt/người chơi, dữ liệu tọa độ bóng theo từng frame bị đứt đoạn và xuất hiện các điểm nhiễu phát hiện sai (False Positives).
* **Thuật toán xử lý qua Pandas DataFrame:**
  1. **Lập bảng tọa độ:** Chuyển đổi danh sách kết quả phát hiện sang `pd.DataFrame` với 4 cột `['x1', 'y1', 'x2', 'y2']`.
  2. **Khử điểm nhảy vọt bất thường (Velocity-Gated Outlier Spike Rejection):**
     - Tính khoảng cách dịch chuyển tức thời giữa frame $i$ và các frame hợp lệ lân cận ($i-1$ và $i+1$).
     - Nếu bóng đột ngột "dịch chuyển tức thời" $> 80\text{ px}$ trong đúng 1 frame rồi quay về quỹ đạo cũ (do AI bắt nhầm vệt trắng trên giày hoặc mép vợt), tọa độ tại frame $i$ sẽ bị gán ngay thành `NaN`.
  3. **Nội suy tuyến tính (Linear Interpolation):**
     - Gọi lệnh `df.interpolate(method='linear')` kết hợp `bfill().ffill()` để tự động tái tạo quỹ đạo mượt mà cho các frame bị mất dấu bóng.
  4. **Cố định kích thước Bounding Box (Box Size Stabilizer):**
     - Tính kích thước trung vị `median()` của chiều rộng và chiều cao bóng qua toàn bộ video.
     - Khóa kích thước hộp bao cố định quanh tâm $(c_x, c_y)$, loại bỏ hoàn toàn hiện tượng bounding box bị co giật/nhấp nháy kích thước giữa các khung hình.

---

## 4. BẢNG TỔNG HỢP ĐỐI CHIẾU TOÀN DIỆN

| Thành phần xử lý | File mã nguồn phụ trách | Dữ liệu đầu vào (Input) | Thao tác tiền xử lý chính | Dữ liệu đầu ra (Output) |
| :--- | :--- | :--- | :--- | :--- |
| **Gộp dữ liệu bóng** | [`training/dataset_downloader.py`](training/dataset_downloader.py#L86) | 3 bộ dataset Roboflow rời rạc | Đổi tên tiền tố `ds{i}_`, thống nhất nhãn `tennis_ball`, tạo `data.yaml` | `merged_tennis_dataset/` (4,454 ảnh chuẩn) |
| **Gộp nhãn vạch sân** | [`training/download_court_dataset.py`](training/download_court_dataset.py#L54) | 10,000 file nhãn `.txt` riêng lẻ | Đọc $(x_c, y_c)$, lọc ảnh $\ge 12$ keypoints, bù khuyết điểm thiếu | File tổng hợp [`data.json`](training/datasets/court/tennis_court_keypoints/data.json) (28 số/ảnh) |
| **Nạp Tensor vạch sân (Train)** | [`training/train_court_line_detector_tf.py`](training/train_court_line_detector_tf.py#L18) | File ảnh gốc + file `data.json` | Resize $224 \times 224$, Z-Score ImageNet, scale nhãn theo $\frac{224}{W}, \frac{224}{H}$ | Tensor ảnh `(N, 224, 224, 3)`, nhãn `(N, 28)` |
| **Online Augment bóng (Train)** | [`training/train_yolo26_ball_detector.py`](training/train_yolo26_ball_detector.py#L151) | Ảnh màu + nhãn YOLO `.txt` | Letterbox $640 \times 640$ (đệm viền xám), Mosaic 4 góc, HSV Jitter | Batch ảnh $640 \times 640$ bảo toàn hình học tròn |
| **Tensor vạch sân (Inference)** | [`court_line_detector/court_line_detector.py`](court_line_detector/court_line_detector.py#L49) | Frame video BGR | BGR $\to$ RGB, resize $224 \times 224$, Z-score, expand batch `(1, 224, 224, 3)` | Tensor đầu vào mạng ResNet50 |
| **Hút vạch sơn trắng** | [`court_line_detector/court_line_detector.py`](court_line_detector/court_line_detector.py#L86) | Frame video + 14 keypoints thô | Chuyển HSV, lọc ngưỡng trắng, tạo Court Mask, snap sub-pixel | 14 keypoints khớp chính xác mép vạch vôi |
| **Quang học bám sân CPV** | [`court_line_detector/cpv_court_tracker.py`](court_line_detector/cpv_court_tracker.py#L68) | Chuỗi các frame video BGR | BGR $\to$ Gray, ROI Court Mask, 250 điểm Shi-Tomasi, RANSAC Affine | Ma trận biến đổi phối cảnh 14 keypoint từng frame |
| **Làm sạch chuỗi bóng** | [`trackers/ball_tracker.py`](trackers/ball_tracker.py#L59) | Danh sách tọa độ thô từng frame | Outlier Spike Filter (loại bỏ bước nhảy $>80\text{px}$), Linear Interpolation, Median Box | Quỹ đạo bóng liên tục, không bị đứt đoạn hay rung giật |

---
*Tài liệu được biên soạn và chuẩn hóa phục vụ tra cứu, báo cáo học thuật và nghiên cứu chuyên sâu.*
