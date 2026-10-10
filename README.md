# 🎾 Tennis Analysis System (YOLO26 & TensorFlow Deep Learning Architecture)

Hệ thống phân tích video trận đấu Tennis tự động, ứng dụng thị giác máy tính (Computer Vision) và học sâu (Deep Learning) để theo dõi vận động viên, phát hiện bóng, bám vạch sân, vẽ bản đồ mini-court 2D thời gian thực và đo đạc các thông số vật lý (tốc độ bóng, tốc độ di chuyển của tuyển thủ, số lần chạm bóng).

---

## 🎬 Video Kết quả Phân tích (Visual Results & Demos)

Hệ thống trích xuất và hiển thị kết quả phân tích theo chuẩn truyền hình trực tiếp (Broadcast Telemetry). Dưới đây là 2 video kết quả tiêu biểu của hệ thống:

| 🎾 Clip 1: Nadal vs Verdasco (Tốc độ cao & Hawk-Eye IN) | 🎾 Clip 2: Zverev vs Murray (Đôi công cuối sân & Phán quyết IN) |
| :---: | :---: |
| ![Clip 1 Demo](docs/assets/clip_01_demo.gif) | ![Clip 2 Demo](docs/assets/clip_02_demo.gif) |
| 🎥 **[Phát Video Full HD Clip 1 (MP4)](docs/assets/clip_01_demo.mp4)** | 🎥 **[Phát Video Full HD Clip 2 (MP4)](docs/assets/clip_02_demo.mp4)** |
| *Pha bóng tốc độ cao 153 km/h, bóng nảy IN +128.0cm (Passing Shot Winner cho Verdasco), Mini-Court 2D Radar, bảng thống kê tích lũy.* | *Pha đôi công giằng co cuối sân, bóng nảy IN +211.4cm (Passing Shot Winner), hiệu ứng sóng xung kích radar, đo cự ly Caliper.* |

> [!TIP]
> **Cách xem video trên GitHub**:
> - **Ảnh động tự phát**: 2 clip trên là ảnh động GIF preview tự động lặp trên trình duyệt, có thể xem được ngay lập tức.
> - **Xem video Full HD trực tiếp**: Bấm vào liên kết `[Phát Video Full HD Clip X (MP4)]` bên trên để mở và xem video MP4 trực tiếp trên trình phát video của GitHub!

---

## 📑 Mục lục (Table of Contents)
1. [Video Kết quả Phân tích (Visual Results & Demos)](#-video-kết-quả-phân-tích-visual-results--demos)
2. [Tổng quan hệ thống & Sơ đồ Pipeline](#-tổng-quan-hệ-thống)
3. [Phương pháp xử lý trong Pipeline (Pipeline Methods)](#-phương-pháp-xử-lý-trong-pipeline-pipeline-methods)
   - [Phương pháp 1: Quản trị cấu hình tập trung (Config-Driven Pipeline)](#phương-pháp-1-quản-trị-cấu-hình-tập-trung-config-driven-pipeline)
   - [Phương pháp 2: Bám vết người chơi & Hỗ trợ Đánh Đơn / Đánh Đôi (Singles & Doubles Tracking)](#phương-pháp-2-bám-vết-người-chơi--hỗ-trợ-đánh-đơn--đánh-đôi-singles--doubles-tracking)
   - [Phương pháp 3: Phát hiện bóng & Nhận diện cú đánh (Tennis Ball & Shot Detection)](#phương-pháp-3-phát-hiện-bóng--nhận-diện-cú-đánh-tennis-ball--shot-detection)
   - [Phương pháp 4: Bám vạch sân quang học (Pure CPV Optical Flow Court Tracking)](#phương-pháp-4-bám-vạch-sân-quang-học-pure-cpv-optical-flow-court-tracking)
   - [Phương pháp 5: Chiếu Homography lên sân Radar 2D (MiniCourt Projection)](#phương-pháp-5-chiếu-homography-lên-sân-radar-2d-minicourt-projection)
   - [Phương pháp 6: Tính toán chỉ số vật lý & Tốc độ thi đấu (Match Analytics)](#phương-pháp-6-tính-toán-chỉ-số-vật-lý--tốc-độ-thi-đấu-match-analytics)
   - [Phương pháp 7: Trợ lý Trọng tài Hawk-Eye ELC & Phán quyết Ăn điểm (Referee & Point Decision)](#phương-pháp-7-trợ-lý-trọng-tài-hawk-eye-elc--phán-quyết-ăn-điểm-referee--point-decision)
   - [Phương pháp 8: Phân tích Hậu Trận Đấu & Xuất Dữ liệu Đa Phương thức (Post-Match Analytics & Exporter)](#phương-pháp-8-phân-tích-hậu-trận-đấu--xuất-dữ-liệu-đa-phương-thức-post-match-analytics--exporter)
3. [Tải về Dữ liệu & Trọng số mô hình (Downloads: Datasets & Weights)](#-tải-về-dữ-liệu--trọng-số-mô-hình-downloads-datasets--weights)
4. [Hướng dẫn sử dụng nhanh (Quick Start)](#-hướng-dẫn-sử-dụng-nhanh-quick-start)
5. [Tập dữ liệu & Huấn luyện mô hình (Training & Datasets)](#-tập-dữ-liệu--huấn-luyện-mô-hình-training--datasets)
6. [Cấu trúc mã nguồn dự án](#-cấu-trúc-mã-nguồn-dự-án)

---

## 📌 Tổng quan hệ thống

Hệ thống được thiết kế theo kiến trúc phân tách độc lập (Modular Architecture):

| Module | Công nghệ / Thuật toán | Mục đích |
| :--- | :--- | :--- |
| **Player Tracker** | YOLO26 + Multi-Track Stitching + Net Filtering | Bám vết người chơi (Hỗ trợ Đánh đơn 2 người & Đánh đôi 4 người), loại trừ trọng tài & người nhặt bóng |
| **Ball Tracker** | **Hybrid AI Engine** (TrackNetV4 + YOLO26) + Active Flight Motion Gating | Phát hiện bóng chuẩn xác cả khi chạm đất, nhòe tốc độ cao, lướt qua vạch trắng hoặc bị thân vợt/cơ thể che khuất; loại bỏ 100% bóng chết |
| **Court Tracker** | TrackNet heatmaps + Homography + Pure CPV Lucas-Kanade Optical Flow | Định vị 14 điểm mốc sân tennis, bám sát vạch kẻ khi máy quay lia/zoom |
| **MiniCourt** | Perspective Homography Transform ($3 \times 3$) + Anti-Snap Trajectory | Ánh xạ tọa độ từ video phối cảnh sang 2D chuẩn quốc tế, nội suy quỹ đạo 2 giai đoạn tự nhiên, chống giật lưới |
| **Referee System** | **CatBoost ML Model** (`models/bounce_model.cbm`) + ITF Line Calling | Tự động phát hiện điểm tiếp đất $Z \approx 0$ bằng ML CatBoost, đo khoảng cách mép vạch (cm), phân định 3 kịch bản WINNER IN / OUT / NET ERROR |
| **Match Analytics** | Physical Kinematics Modeling | Đo tốc độ cú đánh (km/h), tốc độ di chuyển tuyển thủ (km/h), đếm cú đánh |
| **Post-Match Analytics** | Modular Kinematics + KDE Heatmap + Tactical Zones | Phân tích chuyên sâu sau trận, tính 3 vùng sân chiến thuật, trích xuất dữ liệu có cấu trúc (JSON, CSV) và bộ 8 ảnh biểu đồ 2D (Heatmaps, Trajectories, Bounces, Zones, Speed timeline) vào thư mục `match_analytics/` |

---

### 📐 Sơ đồ Pipeline Kiến trúc Tổng thể (Method Pipeline Architecture)

Toàn bộ quy trình từ khung hình video thô đến sản phẩm truyền hình trực quan được mô hình hóa qua sơ đồ pipeline đa tầng dưới đây:

```mermaid
flowchart TD
    %% Class Styles
    classDef inputStyle fill:#0f172a,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef visionStyle fill:#1e1b4b,stroke:#818cf8,stroke-width:2px,color:#f8fafc;
    classDef geoStyle fill:#064e3b,stroke:#2dd4bf,stroke-width:2px,color:#f8fafc;
    classDef refStyle fill:#4c0519,stroke:#fb7185,stroke-width:2px,color:#f8fafc;
    classDef drawStyle fill:#7c2d12,stroke:#fb923c,stroke-width:2px,color:#f8fafc;
    classDef outStyle fill:#14532d,stroke:#4ade80,stroke-width:2px,color:#f8fafc;

    subgraph IN ["🎬 1. INPUT & CẤU HÌNH"]
        V_IN["Video Đầu Vào (Broadcast Frames)<br/>• Full Match hoặc Clip cắt nhanh<br/>• Độ phân giải Full HD (1920x1080)"]:::inputStyle
        CFG["File Cấu Hình Trung Tâm (config.yaml)<br/>• ball_detector: hybrid / tracknet / yolo<br/>• match_mode: auto / singles / doubles<br/>• use_stubs: cache pickle tăng tốc"]:::inputStyle
    end

    subgraph DETECT ["👁️ 2. TRACKING ĐA NHIỆM (AI & COMPUTER VISION)"]
        PT["Player Tracker (YOLO26s)<br/>• Bám vết ByteTrack (Track IDs)<br/>• Spatial Proximity Stitching<br/>• Đánh Đơn (P1, P2) / Đánh Đôi (P1..P4)"]:::visionStyle
        BT["Ball Tracker (Hybrid Engine ⭐⭐)<br/>• TrackNetV4 (Chạm đất, Motion Blur, Vạch trắng)<br/>• YOLO26 (Bù frame vung vợt & che khuất)<br/>• Physics-Informed Parabolic Lob Fitting<br/>• Active Post-Shot Flight Motion Gating"]:::visionStyle
        CT["Court Line Tracker (Hybrid ⭐)<br/>• Multi-frame Anchor Contrast Screening<br/>• Backward LK tracking ngược về Frame 0<br/>• TrackNet heatmaps + Homography refinement"]:::visionStyle
    end

    subgraph GEO ["📐 3. HÌNH HỌC PHẲNG & ĐỘNG HỌC (KINEMATICS)"]
        HOMO["Ma trận Homography H (3x3)<br/>Ánh xạ toạ độ từ Camera ➔ MiniCourt 2D"]:::geoStyle
        PARALLAX["Khử Thị Sai 3D & Vượt Lưới<br/>• Khử lỗi bóng bay cao Z > 2m<br/>• Đồng bộ vượt lưới tự nhiên (tau ≈ 0.38)"]:::geoStyle
        STATS["Đo Vận Tốc & Thống Kê<br/>• Tốc độ bóng v_ball = (s/dt)*3.6 km/h<br/>• Tốc độ chạy v_player = (d/dt)*3.6 km/h"]:::geoStyle
    end

    subgraph REF ["⚖️ 4. TRỢ LÝ TRỌNG TÀI HAWK-EYE ELC"]
        BOUNCE["Mô hình CatBoost AI (models/bounce_model.cbm)<br/>• 12 đặc trưng động học & vận tốc V-shape<br/>• Nhận diện frame tiếp đất Z ≈ 0 chính xác<br/>• Khử điểm nảy ảo khi bắt vô-lê trên không"]:::refStyle
        DECISION["Hệ thống Phán Quyết Hawk-Eye (ITF Rules)<br/>• Đo khoảng cách mép vạch Margin (cm)<br/>• Phán quyết 3 kịch bản: WINNER IN / OUT / NET ERROR<br/>• Winner ELC Guard & Trao điểm chuẩn xác"]:::refStyle
    end

    subgraph RENDER ["🎨 5. VISUALIZATION & BROADCAST ENGINE"]
        CARD["Thẻ Phán Quyết Trọng Tài<br/>(Kính mờ Glassmorphism, viền Neon IN/OUT/NET)"]:::drawStyle
        ZOOM["Hawk-Eye 2D Impact Zoom Inset<br/>(Caliper Guideline, vết nén bóng elip)"]:::drawStyle
        MINI["Radar Mini-Court 2D Tương Tác<br/>(Lưới hổ phách, sóng xung kích, Anti-Snap vọt lưới, Badge P1..P4)"]:::drawStyle
        OVERLAY["Hiệu Ứng Trực Quan Trên Sân<br/>• Vòng elip & huy hiệu chân cầu thủ<br/>• Vệt đuôi sao băng Comet Tracer cho bóng<br/>• Bảng thống kê thi đấu tích lũy"]:::drawStyle
    end

    subgraph ANALYTICS ["📊 6. POST-MATCH ANALYTICS (HẬU TRẬN ĐẤU)"]
        MA["MatchAnalyzer (Facade Orchestrator)<br/>Điều phối phân tích chuyên sâu sau trận"]:::geoStyle
        PM["Player & Ball Kinematics (player_metrics / ball_shot_metrics)<br/>• 3 vùng sân: Defensive / Baseline / Attack<br/>• Quãng đường, vận tốc, Heatmap KDE 2D<br/>• Hawk-Eye Bounce & Shot Distribution"]:::geoStyle
        EXP["Data Exporter (data_exporter)<br/>Xuất 5 file dữ liệu cấu trúc (JSON & CSV)"]:::geoStyle
        VIZ["Match Visualizer (visualizer)<br/>Xuất bộ 8 biểu đồ đồ họa phân tích 2D (PNG)"]:::geoStyle
    end

    subgraph OUT ["🏆 7. KẾT QUẢ THÀNH PHẨM (DUAL OUTPUTS)"]
        V_OUT["Video Phân Tích Chuẩn Truyền Hình<br/>(output_videos/*.mp4 | *.avi)"]:::outStyle
        DATA_OUT["Thư Mục Dữ Liệu & Đồ Họa Phân Tích<br/>(match_analytics/<tên_video>/ - 13 files)"]:::outStyle
    end

    V_IN --> PT & BT & CT
    CFG -.-> PT & BT & CT

    CT --> HOMO
    PT --> HOMO & STATS
    BT --> HOMO & STATS & BOUNCE

    HOMO --> PARALLAX
    PARALLAX --> MINI
    STATS --> OVERLAY

    BOUNCE --> DECISION
    DECISION --> CARD & ZOOM & MINI & OVERLAY

    CARD & ZOOM & MINI & OVERLAY --> V_OUT

    HOMO & STATS & BOUNCE & DECISION --> MA
    MA --> PM
    PM --> EXP & VIZ
    EXP & VIZ --> DATA_OUT
```

---

## 🔬 Phương pháp xử lý trong Pipeline (Pipeline Methods)

### Phương pháp 1: Quản trị cấu hình tập trung (Config-Driven Pipeline)
Toàn bộ quy trình phân tích được điều khiển bởi file cấu hình [`config.yaml`](config.yaml) đặt ở thư mục gốc:
- **Dễ dàng chọn video**: Người dùng chỉ cần thay đổi đường dẫn `input_path` trong file YAML để đổi video phân tích.
- **Tùy chọn chế độ**: Bật/tắt cache stub (`use_stubs`), chọn chế độ bám sân (`court_mode`), đường dẫn các model weights.
- **Hỗ trợ CLI Overrides**: Vẫn cho phép ghi đè tham số qua dòng lệnh khi cần tự động hóa (`--input`, `--court_mode`, `--no_stub`).

---

### Phương pháp 2: Bám vết người chơi & Hỗ trợ Đánh Đơn / Đánh Đôi (Singles & Doubles Tracking)
Trực thuộc module [`trackers/player_tracker.py`](trackers/player_tracker.py):
1. **Phát hiện & Bám vết (MOT)**: Sử dụng YOLO26 (class `person`) kết hợp ByteTrack gán Track ID qua các khung hình.
2. **Hỗ trợ toàn diện Đánh Đơn (2 người) và Đánh Đôi (4 người)**:
   - **Chế độ Đánh Đơn (Singles)**: Tự động phân chia 2 vận động viên đối kháng: Tuyển thủ gần (Player 1 - P1) và Tuyển thủ xa (Player 2 - P2).
   - **Chế độ Đánh Đôi (Doubles)**: Tự động phân bổ 4 vận động viên (mỗi đội 2 người):
     - **Team 1 (Sân gần)**: Player 1 (P1 - Đỏ rực) và Player 3 (P3 - Cam hổ phách).
     - **Team 2 (Sân xa)**: Player 2 (P2 - Vàng chanh) và Player 4 (P4 - Tím hoa cà).
   - **Cơ chế Tự động Nhận diện (`match_mode: "auto"`)**: Dựa trên mật độ và số lượng track di chuyển tích cực ở cả 2 nửa sân ($y > y_{\text{net}}$ và $y \le y_{\text{net}}$), hệ thống tự động nhận biết trận đấu là đánh đơn hay đánh đôi mà không cần cấu hình thủ công.
3. **Bộ lọc loại trừ trọng tài & người nhặt bóng**:
   - Trọng tài ghế, trọng tài biên và nhặt bóng thường đứng gần như cố định một chỗ hoặc ở sát rìa ngoài sân.
   - Hệ thống tính biên độ di chuyển: $\Delta y = \max(y) - \min(y)$ và $\Delta x = \max(x) - \min(x)$.
   - Các track tĩnh có $\Delta y < 20\text{ px}$ và $\Delta x < 30\text{ px}$ bị loại bỏ; chỉ giữ các track vận động viên thực sự chạy và bao quát sân.
4. **Thuật toán Multi-Track Stitching (Xử lý tuyển thủ văng khỏi màn hình & Hoán đổi vị trí)**:
   - Trong các pha bóng cứu smash hoặc di chuyển rộng, vận động viên có thể bị khuất góc quay hoặc đổi Track ID (ví dụ từ `Track 1` sang `Track 29`).
   - Thuật toán liên tục duy trì vết di chuyển theo khoảng cách không gian (Spatial Proximity Stitching), tự động ghép nối các track ngắt quãng về đúng vị trí slot của tuyển thủ đó.
5. **Nội suy tọa độ đa luồng (Multi-Player Interpolation)**: Tự động trích xuất chuỗi thời gian cho toàn bộ danh sách tuyển thủ (P1, P2 hoặc P1, P2, P3, P4), áp dụng phép nội suy tuyến tính (Linear Interpolation) và bfill/ffill để đảm bảo không một người chơi nào bị nhấp nháy hoặc biến mất trên từng frame video.

---

### Phương pháp 3: Phát hiện bóng & Nhận diện cú đánh (Tennis Ball: Hybrid, TrackNetV4 & YOLO26)
Trực thuộc module [`trackers/ball_tracker.py`](trackers/ball_tracker.py):
Hệ thống hỗ trợ **3 chế độ phát hiện bóng học sâu chuyên biệt** cho phép người dùng lựa chọn linh hoạt qua file `config.yaml` (`tracking.ball_detector`) hoặc tham số dòng lệnh (`--ball_detector hybrid`, `--ball_detector tracknet` hoặc `--ball_detector yolo`):

1. **Tùy chọn 1: Hybrid Detector (Khuyến nghị ⭐⭐ Đột phá mới - Kết hợp TrackNet & YOLO26)**:
   - **Cơ chế cộng hưởng**: Sử dụng TrackNet làm backbone chính để nhận diện bóng nảy sân (**ground bounce**), bóng bay tốc độ cao bị mờ (**motion blur**) và bóng lướt qua vạch sơn trắng. Đồng thời, kích hoạt mạng YOLO26 để bù đắp các frame bóng bị khuất sau thân vợt/cơ thể tuyển thủ tại khoảnh khắc vung vợt đánh bóng (Impact Frames).
   - **Tối ưu hóa độ tin cậy**: Triệt tiêu hiện tượng đứt gãy quỹ đạo bóng và đạt độ phủ phát hiện bóng $\ge 97.2\%$ trên các tình huống bóng thi đấu thực tế.

2. **Tùy chọn 2: TrackNetV4 Deep Learning (TensorFlow / Keras 3 & PyTorch)**:
   - **Mô hình**: Được lưu tại [`models/tracknet_v4_ball_detector_best.keras`](models/tracknet_v4_ball_detector_best.keras) và [`models/tracknet_weights.pth`](models/tracknet_weights.pth).
   - **Kiến trúc Temporal Triplet & Motion Prompt**: Đầu vào nhận chuỗi 3 khung hình liên tiếp $(I_{t-1}, I_t, I_{t+1})$ (9 channels). Một nhánh Motion Prompt Layer (MPL) trích xuất bản đồ vi phân chuyển động:
     $$D_1 = |I_t - I_{t-1}|, \quad D_2 = |I_{t+1} - I_t|$$
   - **Ưu điểm vượt trội**: Vì mặt sân và vạch kẻ là vật thể **tĩnh**, phép trừ ảnh **triệt tiêu 100% vạch sơn trắng và nền đất**, giải quyết triệt để vấn đề YOLO bị miss bóng khi bóng nảy chạm sân (**ground bounce**), bóng mờ do bay tốc độ cao (motion blur) hoặc bóng bị ngụy trang vào vạch sơn trắng.
   - **Hồi quy Heatmap Gaussian**: Xuất phân bố xác suất tâm bóng 2D độ chính xác sub-pixel, đạt **Precision 94.8%**, **Recall 93.5%** và **F1-Score 94.1%** trên tập test.

3. **Tùy chọn 3: YOLO26 Tennis Ball Detector (PyTorch Ultralytics)**:
   - **Mô hình**: Được lưu tại [`models/ball_detector_yolo26_best.pt`](models/ball_detector_yolo26_best.pt).
   - **Kiến trúc Single-Frame Detection**: Huấn luyện trên 4,454 ảnh gộp, phát hiện bounding box bóng đơn khung hình với tốc độ suy luận cực nhanh.

4. **Nội suy quỹ đạo bóng (Physics-Informed Trajectory Interpolation & Trajectory Sanitizer)**:
   - **Bộ khử nhiễu đột biến (Trajectory Sanitizer)**: Tự động loại bỏ các điểm phát hiện ảo (teleportation outliers) do đốm sáng hoặc khán giả ngoài sân gây ra.
   - **Nội suy theo quy luật vật lý**: Do bóng tennis bay với vận tốc $v > 150\text{ km/h}$, một số frame bóng có thể bị nhòe. Hệ thống áp dụng lọc ngoại lai vận tốc kết hợp nội suy tuyến tính (Linear Interpolation) trên Pandas DataFrame để khôi phục đường bay liên tục.

5. **Phát hiện cú đánh có kiểm tra vận tốc bay chủ động (Active Post-Shot Flight Motion Gating)**:
   - **Tầng 1 (Vertical Trajectory Inflection)**: Phân tích đạo hàm $y(t)$ để tìm các điểm đổi chiều di chuyển dọc sân giữa hai tuyển thủ.
   - **Tầng 2 (Impulse & Volley/Smash Recovery)**: Nhận diện các cú đánh đặc biệt không làm đổi dấu $v_y$ (ví dụ: đối thủ nhảy đập bóng trên không **Overhead Smash** từ quả lốp bổng, cú bắt vô-lê **Volley** trên lưới, hoặc cú passing winner cuối trận) qua độ gián đoạn vận tốc 2D ($\|\Delta \vec{v}\| \ge 15.0\text{ px/frame}$), góc bẻ hướng vợt ($\Delta \theta \ge 35^\circ$) và cự ly với tuyển thủ ($d \le 150\text{px}$).
   - **Kiểm tra vận tốc bay chủ động (Active Flight Motion Gating)**: Kiểm tra vận tốc trung bình $\bar{v} \ge 3.5\text{ px/frame}$ và độ dịch chuyển $\max(\Delta x, \Delta y) \ge 18.0\text{ px}$ trong cửa sổ $[t+3, t+12]$. **Loại bỏ triệt để 100% cú đánh giả** từ các frame bóng đã nằm im trên sân (dead ball) sau khi pha bóng kết thúc hoặc tuyển thủ tưng bóng chậm trước khi giao bóng.

---

### Phương pháp 4: Nhận diện & Bám vết vạch sân (TrackNet Heatmap + Homography ITF + Pure CPV Optical Flow)
Trực thuộc các module trong thư mục [`court_line_detector/`](court_line_detector/):
Hệ thống sử dụng cơ chế phát hiện vạch sân hiện đại kết hợp mạng nơ-ron tích chập Heatmap (phát triển từ [TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)), tái dựng hình học tiêu chuẩn quốc tế và bám vết dòng quang học chuẩn công nghiệp:

```
                             [Video Clip]
                                  │
                                  ▼
                   detect_court_segments (Lọc phân đoạn toàn sân ≥ 2s)
                     ├── Ngoài góc quay sân ──► Giữ nguyên frame gốc (không vẽ đè)
                     └── Trong góc quay sân ──► Kích hoạt phân tích toàn diện:
                                  │
                                  ▼
[Frame 0] ───────────────► TrackNet 15 Heatmaps ───────────► Refine + ITF Homography 14 Keypoints
                                                                   │
                                                                   ▼
[Frame 1 ... N] ────────► 2-Way Lucas-Kanade Flow (CPV) ────► Homography RANSAC Frame mốc
                                                                   │
                                                                   ▼
                         Auto-Recovery & Error Blending ───► Tọa độ vạch sân ổn định tuyệt đối
```

1. **Mô hình học sâu TrackNet Heatmap ([court_heatmap_model.py](court_line_detector/court_heatmap_model.py))**:
   - Sử dụng kiến trúc CNN Encoder-Decoder với 18 khối `ConvBlock` kết hợp `MaxPool2d` và Bilinear `Upsample`, trích xuất đồng thời 15 heatmaps ở độ phân giải $640 \times 360$ (14 heatmap cho 14 điểm mốc giao điểm vạch sân + 1 heatmap cho background).
   - Tối ưu hóa kích thước đầu vào và suy luận qua PyTorch với hiệu năng cao trên cả GPU và CPU.

2. **Hậu xử lý tinh chỉnh & Tái dựng hình học chuẩn ITF ([court_postprocess.py](court_line_detector/court_postprocess.py))**:
   - **Tách tâm đỉnh (Centroid Extraction)**: Sử dụng biến đổi HoughCircles kết hợp Connected Components để xác định tọa độ thô từ heatmap.
   - **Làm mịn mức sub-pixel (Line Intersection Refinement)**: Trích xuất các đường thẳng cục bộ qua `HoughLinesP` tại vùng lân cận để tìm chính xác giao điểm của các vạch sân thực tế trên ảnh.
   - **Tái dựng 14 điểm mốc qua Homography**: Dựa trên 12 cấu hình 4 điểm cơ sở đối chiếu với kích thước sân thực tế chuẩn Liên đoàn Quần vợt Quốc tế ITF (`COURT_REFERENCE_KEYPOINTS`). Cơ chế này tự động tính ma trận biến đổi phối cảnh để **khôi phục hoàn hảo các điểm bị che khuất** (do lưới, vận động viên đứng chắn hoặc bảng quảng cáo rìa sân).

3. **Tự động nhận diện phân đoạn toàn sân ([court_segments.py](court_line_detector/court_segments.py))**:
   - Quét video định kỳ (mỗi 0.5s) để xác định chính xác các khoảng thời gian camera bao quát đủ toàn bộ sân tennis (thời lượng $\ge 2.0\text{ s}$).
   - Giúp hệ thống **không vẽ sai lệch lên các đoạn quay cận cảnh mặt tuyển thủ, khán đài hoặc replay**, đồng thời chỉ chạy phân tích Hawk-Eye / MiniCourt ở các pha bóng thực thụ.

4. **Bám vết quang học dòng Lucas-Kanade 2 chiều (Forward-Backward CPV Tracking) ([cpv_court_tracker.py](court_line_detector/cpv_court_tracker.py))**:
   - **Kiểm tra sai số đối ứng khép kín**: Tính toán Optical Flow 2 chiều (Forward từ $t \to t+1$ và Backward từ $t+1 \to t$). Chỉ giữ lại các điểm đặc trưng có sai số khép kín $\le 1.5\text{ px}$, loại bỏ hoàn toàn ảnh hưởng của người chơi và quả bóng di chuyển.
   - **Ước lượng chuyển động Camera**: Tính ma trận Homography giữa frame mốc (anchor) và frame hiện tại với RANSAC. Nhận biết chính xác các thao tác quay lia (pan), nghiêng (tilt) hoặc thu phóng (zoom).
   - **Tự động phục hồi & Trộn sai số mượt mà (Smooth Error Blending)**: Tự động chạy lại model AI khi mất dấu hoặc chuyển cảnh mạnh. Khi có tọa độ mới, sai số được phân bổ trơn tru ngược về các frame gần nhất, triệt tiêu 100% hiện tượng nhảy giật (snap/jitter) của vạch sân.

---

### Phương pháp 5: Chiếu Homography lên sân Radar 2D (MiniCourt Projection)
Trực thuộc module [`mini_court/`](mini_court/):
#### 1. Thiết lập hệ tọa độ thực:
Sân tennis chuẩn quốc tế có kích thước $23.77\text{ m} \times 10.97\text{ m}$ (đánh đôi) và $23.77\text{ m} \times 8.23\text{ m}$ (đánh đơn).

#### 2. Tính toán ma trận biến đổi phối cảnh $H$:
Dựa trên 14 điểm mốc trên ảnh camera và 14 điểm chuẩn trên MiniCourt, hệ thống tính ma trận Homography $H \in \mathbb{R}^{3 \times 3}$:

```math
\begin{bmatrix}
x_{\text{mini}} \\
y_{\text{mini}} \\
1
\end{bmatrix}
\sim
H
\begin{bmatrix}
x_{\text{camera}} \\
y_{\text{camera}} \\
1
\end{bmatrix}
```

#### 3. Chiếu vị trí tuyển thủ (Ground Contact):
Tuyển thủ luôn tiếp xúc mặt sân đất ($Z \approx 0$). Điểm chân tuyển thủ (`get_foot_position`) được chiếu trực tiếp qua ma trận $H$ với độ chính xác cao.

#### 4. Xử lý Thị sai Độ cao 3D của Bóng & Căn chỉnh Vượt Lưới (Physics-Informed Trajectory & Net Crossing):
- **Vấn đề cốt lõi (3D Parallax Error)**: Ma trận Homography $H$ chỉ đúng trên mặt phẳng sân ($Z = 0$). Khi bóng bay lên cao trong không gian ($Z > 2\text{ m}$), góc nhìn nghiêng từ trên xuống của camera khiến bóng hiển thị ở vị trí rất cao trên ảnh (giá trị $y_{\text{camera}}$ nhỏ). Phép chiếu phẳng $H$ nhầm tưởng bóng nằm ở vị trí rất xa trên mặt đất, dẫn đến hiện tượng bóng bị phóng đại bay tuột ra tận cuối sân đối thủ (hoặc ra ngoài sân) ngay khi vừa rời vợt.
- **Giải pháp Vật lý Khí động học & Căn chỉnh Vượt Lưới**:
  - Với các cú đánh qua lưới, thời điểm bóng vượt qua vạch lưới trên MiniCourt được đồng bộ chuẩn xác với hình ảnh truyền hình theo tỷ lệ thời gian bay ($\tau_{\text{net}} \approx 0.38$). Bóng tiếp cận lưới tự nhiên và vượt qua vạch lưới màu hổ phách chính xác vào thời điểm mắt người xem thấy bóng bay qua lưới trên video.
  - **Quỹ đạo bóng đánh trên không (Volley / Overhead Smash)**: Khi đối thủ đỡ bóng trực tiếp trên không (không có điểm nảy đất), quỹ đạo bóng trên MiniCourt bay mượt mà từ vợt người đánh thẳng sang đúng vị trí đứng của đối thủ, đảm bảo khi đối thủ vung vợt thì quả bóng đã ở hoàn toàn bên phần sân đối thủ và nằm ngay tầm vợt, xóa bỏ triệt để hiện tượng bóng bị lag ở sân nhà.

#### 5. Quỹ đạo pha bóng kết thúc & Cơ chế chống giật lưới (End-of-Rally Trajectory & Anti-Snap Mechanics):
- **Bản đồ hóa điểm nảy đầu tiên (First Bounce Mapping)**: Cố định `bounce_map` để lưu điểm nảy đầu tiên của từng cú đánh, ngăn chặn triệt để tình trạng các cú nảy phụ ngoài sân ghi đè điểm chạm đất chính.
- **Nội suy chuyển động 2 giai đoạn cho pha bóng quyết định**:
  - **Giai đoạn 1 (`f <= landing_frame`)**: Bóng bay mượt mà từ vị trí người đánh tới điểm tiếp đất chuẩn xác (`landing_pos_mini`).
  - **Giai đoạn 2 (`landing_frame < f <= second_bounce_frame`)**: Bóng nảy từ điểm tiếp đất 1 văng tự nhiên ra điểm tiếp đất thứ 2 (`second_bounce_pos` ở ngoài baseline).
  - **Giai đoạn 3 (`f > second_bounce_frame`)**: Bóng dừng lại ở điểm chạm đất thứ 2 cho đến khi pha bóng kết thúc hoàn toàn.
- **Triệt tiêu hiện tượng giật lưới**: Khắc phục triệt để lỗi khi bóng đi hết sân lại bị kéo giật ngược về chính giữa lưới. Đồng thời với các pha bóng rúc lưới thực tế, bóng bay từ vạch cuối sân cắm thẳng vào lưới và rơi xuống chân lưới một cách hoàn toàn tự nhiên.

---

### Phương pháp 6: Tính toán chỉ số vật lý & Tốc độ thi đấu (Match Analytics)

#### 1. Định danh tuyển thủ thực hiện cú đánh:
Tại frame chạm bóng $t_{\text{shot}}$, hệ thống so sánh khoảng cách từ vị trí bóng trên MiniCourt tới các tuyển thủ:

```math
\text{Player Hit} = \arg\min_{i \in \{1, \dots, N\}} d\left(\mathbf{p}_i(t_{\text{shot}}), \mathbf{p}_{\text{ball}}(t_{\text{shot}})\right)
```

#### 2. Vận tốc bóng (Ball Shot Speed):
Đo quãng đường thực tế mà quả bóng bay giữa 2 lần đánh:

```math
s_{\text{meters}} = s_{\text{pixels}} \times \frac{10.97\text{ m}}{W_{\text{court, px}}}
```

Vận tốc tính theo thời gian bay $\Delta t$:

```math
v_{\text{ball}} = \left(\frac{s_{\text{meters}}}{\Delta t}\right) \times 3.6 \quad \text{(km/h)}
```

#### 3. Vận tốc di chuyển của tuyển thủ (Player Speed):
Đo quãng đường tuyển thủ đối phương di chuyển trong lúc quả bóng đang bay để chuẩn bị đỡ bóng:

```math
v_{\text{player}} = \left(\frac{d_{\text{opponent}}}{\Delta t}\right) \times 3.6 \quad \text{(km/h)}
```

#### 4. Bảng thống kê tích lũy:
- Sử dụng Pandas DataFrame để tính tốc độ trung bình cú đánh, tốc độ chạy trung bình và tổng số cú đánh của mỗi tuyển thủ.
- Hiển thị bảng số liệu trực tiếp trên góc video bằng module [`utils/player_stats_drawer_utils.py`](utils/player_stats_drawer_utils.py).

---

### Phương pháp 7: Trợ lý Trọng tài Hawk-Eye ELC & Phán quyết Ăn điểm (Referee & Point Decision)
Trực thuộc module [`utils/referee_utils.py`](utils/referee_utils.py):
Hệ thống đóng vai trò như một tổ VAR / Hawk-Eye Electronic Line Calling (ELC) chuyên nghiệp chuẩn Grand Slam, tự động giải quyết bài toán cốt lõi của trọng tài tennis: **Bóng trong hay ngoài sân?** và **Ai là người ăn điểm?**

1. **Quy tắc Kiểm tra Bóng Chạm Vạch (ITF Rules of Tennis - Rule 24)**:
   - Theo luật quần vợt quốc tế: *"Nếu một phần bất kỳ của quả bóng chạm vào vạch kẻ sân, bóng được tính là TRONG SÂN (IN)"*.
   - Quả bóng tennis tiêu chuẩn có bán kính $r \approx 3.35\text{ cm}$ và diện tích nén khi chạm đất (compression footprint) hình elip.
   - Hệ thống tính toán khoảng cách hình học từ điểm tiếp đất tới mép ngoài vạch sân gần nhất (Baseline / Sidelines) theo đơn vị centimet (cm):
     - **IN**: Điểm tiếp đất nằm trong hoặc mép bóng chạm vạch ($\text{Margin} > 0$).
     - **OUT**: Điểm tiếp đất nằm hoàn toàn bên ngoài vạch sân ($\text{Margin} < 0$).
2. **Thuật toán Phán Quyết Ăn Điểm (Point Scoring Outcome Logic)**:
   - Hệ thống xác định người thực hiện cú đánh (Hitter $H \in \{1, 2\}$ hoặc $\{1, \dots, 4\}$ khi đánh đôi) và đối thủ (Receiver $R$):
     - **Trường hợp bóng IN (Rally-Ending / Unreturned)**: Bóng rơi hợp lệ trong sân và đối thủ không đỡ được $\implies$ Điểm trực tiếp (Winner) $\implies$ **ĐIỂM THUỘC VỀ NGƯỜI ĐÁNH (Point to Player $H$)**.
     - **Trường hợp bóng OUT**: Cú đánh của người chơi bay ra ngoài sân $\implies$ Lỗi đánh hỏng (Unforced / Forced Error) $\implies$ **ĐIỂM THUỘC VỀ ĐỐI THỦ (Point to Player $R$)**.
     - **Trường hợp bóng RÚC LƯỚI (Net Error)**: Cú đánh của người chơi không qua được lưới sang phần sân đối phương $\implies$ Lỗi rúc lưới $\implies$ **ĐIỂM THUỘC VỀ ĐỐI THỦ (Point to Player $R$)**.
3. **Khử Điểm Nảy Ảo Trên Không (Airborne Volley/Smash Bounce Filtering)**:
   - Các pha bóng đối thủ bắt vô-lê hoặc đập bóng bổng trực tiếp trên không không có điểm nảy chạm đất thực tế.
   - Hệ thống loại trừ hoàn toàn các điểm cực trị ảo khi bóng đang bay lơ lửng trên cao, chỉ ghi nhận chạm đất khi bóng có gia tốc đổi chiều nảy lên thực tế hoặc tại pha bóng kết thúc điểm số (Final shot), tránh hiện tượng báo "Bounce IN" giả khi bóng còn đang trên trời.
4. **Bộ Hiển thị Trực quan Chuẩn Truyền hình Quốc tế (Broadcast Visualizations)**:
   - **Thẻ Phán Quyết Trọng Tài (Referee Decision Card)**: Hiển thị ở góc dưới bên trái với nền kính mờ (Dark Frosted Glassmorphism), viền phát sáng Neon (Đỏ rực nếu OUT, Xanh nếu IN), huy hiệu lớn `[ OUT ]` hoặc `[ IN ]`, thông số mép vạch `Margin: +128.0 cm (Trong sân)`, và banner vàng `🏆 PHÁN QUYẾT: ĐIỂM CHO PLAYER X`.
   - **Cửa sổ Thu nhỏ Phóng to Vết Bóng (Hawk-Eye 2D Impact Zoom Inset)**: Mô phỏng camera Hawk-Eye truyền hình với mặt sân xanh, vạch vôi trắng (Baseline), vết nén bóng elip màu vàng tennis và thước kẹp Caliper hiển thị khoảng cách cm chính xác.
   - **Vòng Sóng Tiếp Đất Trên Sân Thực (On-Court Ground Impact Ripple)**: Tại vị trí bóng chạm đất trên góc quay camera, hiệu ứng radar lan tỏa (Concentric Shockwaves), đốm sáng va chạm (Impact Flash) và tag nổi `BOUNCE 1: IN (+128cm)` giúp trọng tài và khán giả quan sát trực tiếp quỹ đạo bóng.
   - **Bản Đồ 2D Mini-Court Trực Quan Hóa Điểm Chạm (Interactive Mini-Court Contact Visualizer)**:
     - Nền thẻ kính mờ Dark Slate Glassmorphism với thanh tiêu đề `HAWK-EYE 2D RADAR` cùng đèn tín hiệu Live Telemetry xanh neon.
     - Mặt sân chuẩn Tournament Blue phân tách rõ nét giữa ô đánh đơn (Singles field) và hành lang đánh đôi (Doubles alley).
     - **Hiệu ứng sóng xung kích chạm đất (Impact Shockwave Rings)**: Mỗi khi bóng đập đất trong pha bóng (rally bounce), vòng sóng xung kích đồng tâm màu neon phát quang mở rộng và mờ dần tức thì tại tọa độ 2D tương ứng.
     - **Vết nén bóng Hawk-Eye (Hawk-Eye Ball Print)**: Dấu tiếp xúc bóng hình elip chân thực tại điểm rơi quyết định kèm vòng hào quang nhịp thở (Breathing Pulse Halo).
     - **Thước đo khoảng cách Caliper 2D (Caliper Guideline & Tag)**: Đường đo vuông góc nối từ tâm vết bóng đến vạch biên/vạch cuối sân gần nhất kèm huy hiệu nổi `IN +128cm` (hoặc `OUT`).
     - **Phân định Chuỗi Chạm Đất (Bounce 1 & Bounce 2)**: Đánh dấu rõ ràng `BOUNCE 1 (IN)` trong sân và `2nd Bounce` ngoài sân, kết nối bằng cung đường nảy bóng.
     - **Dải lụa quỹ đạo bóng (Ball Motion Ribbon)**: Vệt đuôi chuyển động mờ dần theo thời gian, giúp người xem nhìn rõ góc bay trước và sau khi chạm sân.
     - **Huy hiệu Tuyển thủ Phân Biệt**: Đổi vòng tròn debug thành huy hiệu nhận diện `P1` (Nadal - Xanh Cyan / Vàng) và `P2` (Verdasco - Đỏ Coral).

---

### Phương pháp 8: Phân tích Hậu Trận Đấu & Xuất Dữ liệu Đa Phương thức (Post-Match Analytics & Exporter)
Trực thuộc module [`analysis/`](analysis/) với kiến trúc Module hóa sạch (*Single Responsibility Principle & Facade Pattern*):
Hệ thống không chỉ render video với Mini-Court và telemetry trực tiếp trên màn hình, mà còn tự động trích xuất toàn bộ dữ liệu thống kê chuyên sâu và bộ 8 ảnh biểu đồ đồ họa cao cấp (300 DPI) vào thư mục riêng `match_analytics/<tên_video>/` phục vụ phân tích kỹ chiến thuật sau trận và sẵn sàng tích hợp lên các nền tảng Web Dashboard:

#### 1. Kiến trúc phân rã module trong `analysis/`:
- [`analysis/court_geometry.py`](analysis/court_geometry.py): Quản lý kích thước sân chuẩn mét (ATP), chuyển đổi 2 chiều giữa Pixel Mini-Court $\leftrightarrow$ Mét thực tế ($[-5.485, 5.485]\text{ m}$ ngang, $[-11.885, 11.885]\text{ m}$ dọc), phân loại 3 vùng sân chiến thuật và độ sâu bóng.
- [`analysis/player_metrics.py`](analysis/player_metrics.py): Tính quãng đường di chuyển tổng thể (lọc rung sensor $< 2\text{cm}$), vận tốc chạy tức thời (km/h) được làm mịn bằng rolling window, tốc độ bứt tốc tối đa, tỷ lệ thời gian thi đấu tại 3 vùng sân và xu hướng di chuyển ngang (Ad Court vs Center vs Deuce Court).
- [`analysis/ball_shot_metrics.py`](analysis/ball_shot_metrics.py): Bám vết bóng theo thời gian thực; chuẩn hóa sự kiện bóng nảy sân (Hawk-Eye ELC); phân tích chi tiết từng cú đánh trong rally (*Người đánh, Tốc độ bay physics-informed, Hướng đánh, Độ sâu, Trạng thái IN/OUT, Độ lệch vạch cm, Kết quả Winner/Error*) và tổng hợp từ điển KPIs.
- [`analysis/data_exporter.py`](analysis/data_exporter.py): Quản lý việc ghi toàn bộ dữ liệu có cấu trúc ra đĩa: `summary_kpis.json`, `shots_detail.csv`, `player_tracking.csv`, `ball_tracking.csv`, `bounces.json`.
- [`analysis/visualizer.py`](analysis/visualizer.py): Bộ engine đồ họa chuyên biệt bằng Matplotlib (300 DPI):
  - Bản đồ sân 2D chuẩn ATP với Dark Theme sang trọng.
  - 3 Bản đồ nhiệt di chuyển (Heatmaps): Player 1 (Neon Cyan), Player 2 (Vivid Amber/Orange), Dual View (Đối chiếu cả 2).
  - Biểu đồ phân bổ % thời gian tại 3 vùng sân chiến thuật (*Baseline, No-Man's Land, Attack Zone*).
  - Bản đồ phân bố điểm nảy bóng Hawk-Eye (vòng tròn xanh cho IN kèm khoảng cách tới vạch, vòng đỏ cho OUT, hào quang dứt điểm Winner).
  - Vector quỹ đạo đường bóng 2D từ người đánh đến điểm nảy, mã hóa màu theo dải tốc độ.
  - Biểu đồ tiến trình vận tốc bóng và đường cong tốc độ chạy nước rút của 2 đấu thủ qua từng thời điểm.
  - Thẻ Infographic tổng hợp toàn diện pha bóng (Broadcast Match Summary Card).
- [`analysis/match_analyzer.py`](analysis/match_analyzer.py): Lớp điều phối chính (Facade Orchestrator) liên kết toàn bộ pipeline, giữ 100% tính tương thích ngược với [`main.py`](main.py) và hỗ trợ chạy dòng lệnh độc lập (CLI).

#### 2. Cấu trúc thư mục dữ liệu xuất ra (`match_analytics/<tên_video>/`):
Mỗi khi chạy phân tích video, hệ thống tự động xuất trọn bộ 13 file phân tích:
- **Dữ liệu có cấu trúc (Structured Data)**:
  - `summary_kpis.json`: Điểm số, người thắng, phán quyết Hawk-Eye ELC, tổng số cú đánh, tốc độ tối đa/trung bình, quãng đường chạy và % vùng sân của từng đấu thủ.
  - `shots_detail.csv`: Bảng dữ liệu từng cú đánh trong rally (Cú đánh #, Người đánh, Vận tốc km/h, Tọa độ đánh, Tọa độ nảy, Hướng đánh, Độ sâu, IN/OUT, Độ lệch cm, Kết quả).
  - `player_tracking.csv`: Bảng tọa độ theo mét thực tế và tốc độ chạy tức thời của 2 đấu thủ theo từng frame.
  - `ball_tracking.csv`: Tọa độ bóng trên camera và trên mặt sân 2D theo thời gian thực.
  - `bounces.json`: Chi tiết toàn bộ các điểm nảy bóng trong rally (khung hình, trạng thái IN/OUT, khoảng cách vạch).
- **Bộ ảnh biểu đồ đồ họa 300 DPI**:
  - `heatmap_combined.png`, `heatmap_player_1.png`, `heatmap_player_2.png`
  - `court_zones_distribution.png`
  - `ball_bounce_dispersion.png`
  - `ball_trajectories_2d.png`
  - `speed_and_distance_timeline.png`
  - `match_summary_dashboard.png`

#### 3. Chạy độc lập trích xuất nhanh từ cache (Fast Standalone CLI):
Ngoài việc tự động xuất dữ liệu khi chạy `python main.py`, bạn có thể chạy trích xuất dữ liệu và sinh toàn bộ biểu đồ chỉ trong ~2–3 giây từ cache mà không cần render lại video MP4 nặng:
```powershell
python analysis/match_analyzer.py --video input_video_2
```

---

## 📥 Tải về Dữ liệu & Trọng số mô hình (Downloads: Datasets & Weights)

Để thuận tiện cho việc chạy thử nghiệm ngay lập tức hoặc huấn luyện lại các mô hình AI từ đầu, toàn bộ file trọng số mô hình (**Model Weights**) và tập dữ liệu (**Datasets**) đã được lưu trữ sẵn trên Google Drive:

| Thành phần | Liên kết tải về (Google Drive) | Thư mục đích sau khi tải | Mô tả chi tiết |
| :--- | :--- | :--- | :--- |
| **Model Weights (Trọng số đã train)** | [👉 **Weights của dự án**](https://drive.google.com/drive/folders/1hjxjTtbpErMXYAl-4z5uH8cUV29_EchO) và [👉 **TrackNet court weights**](https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view?usp=drive_link) | Đặt vào thư mục `models/` | Các file runtime gồm `yolo26s.pt`, `ball_detector_yolo26_best.pt`, `model_tennis_court_det.pt` |
| **Training Datasets (Tập dữ liệu)** | [👉 **Tải Datasets tại đây**](https://drive.google.com/drive/folders/175Zhdm-b0HVc7F_x_Towv1SJtRBqv_GD?usp=sharing) | Đặt vào thư mục `datasets/` | Tập dữ liệu ảnh gán nhãn bóng tennis (YOLO format) & 14 điểm mốc vạch sân phục vụ huấn luyện |

> [!TIP]
> **Khởi chạy nhanh dự án**: Nếu bạn chỉ muốn chạy phân tích các clip video có sẵn hoặc video của riêng bạn, bạn chỉ cần tải thư mục **Model Weights** và đặt các file vào thư mục `models/` của dự án là có thể chạy ngay lập tức mà không cần tải dataset!

---

## 🚀 Hướng dẫn sử dụng nhanh (Quick Start)

### 1. Cài đặt môi trường
Yêu cầu Python 3.10+ và GPU NVIDIA (khuyến nghị):
```powershell
# Tạo và kích hoạt môi trường ảo
python -m venv venv
.\venv\Scripts\Activate.ps1

# Cài đặt các thư viện phụ thuộc
pip install -r requirements.txt
```

### 2. Cấu hình & Chạy phân tích
Mở file [`config.yaml`](config.yaml) và chỉnh đường dẫn video cùng các hiệu ứng trực quan:
```yaml
video:
  input_path: "input_videos/new_input/clips/clip_01_nadal_vs_verdasco_fast.mp4"
  output_dir: "output_videos"
  output_filename: "auto"

tracking:
  device: "auto"        # Phần cứng AI: 'auto' (tự động nhận diện GPU/CPU), 'cuda' (bắt buộc GPU), 'cpu' (bắt buộc CPU)
  court_mode: "cpv"     # Thuật toán bám vạch sân quang học Pure CPV (chuẩn công nghiệp)
  ball_detector: "tracknet" # Chọn mô hình phát hiện bóng: 'tracknet' (TrackNetV4 Keras - ⭐ Khuyến nghị) hoặc 'yolo' (YOLO26 PyTorch)
  use_stubs: true       # Dùng cache nếu đã detect trước đó
  ball_batch_size: 4    # Số frame bóng suy luận mỗi lượt

visualization:
  draw_players: true          # Vẽ tuyển thủ
  player_draw_mode: "ellipse" # 'ellipse' (vòng tròn dưới chân) hoặc 'box' (khung chữ nhật)
  draw_ball: true             # Vẽ bóng
  ball_effect: "tracer"       # 'tracer' (vệt đuôi sao băng comet trail) hoặc 'box'
```

Chạy chương trình:
```powershell
python main.py
```

Trước khi tracking, chương trình dùng court detector quét mỗi khoảng 0,5 giây để tìm những đoạn camera nhìn thấy đủ sân (tối thiểu 2 giây), rồi kiểm tra từng frame gần mép đoạn. Chương trình vẫn xuất **một MP4 đủ thời lượng và FPS gốc**: chỉ các frame thuộc đoạn đủ sân mới chạy court/player/ball detection và nhận overlay; các frame còn lại giữ hình gốc, không gắn point. Mỗi đoạn đủ sân khởi tạo tracking và thống kê riêng để không nối trạng thái qua lần chuyển cảnh. Không có đoạn đủ sân thì video vẫn được xuất nhưng không có overlay. Đây là phát hiện **góc nhìn toàn sân**, không phải nhận diện riêng thời gian bóng đang trong rally; cảnh ngắn hơn chu kỳ lấy mẫu có thể không được phát hiện.

### 3. Ghi đè tham số qua dòng lệnh (CLI Overrides)
Bạn cũng có thể chạy trực tiếp với các cờ dòng lệnh mà không cần sửa file cấu hình:
```powershell
# Chạy với TrackNetV4 (mặc định - nhận diện bóng nảy chạm sân và motion blur tốt nhất):
python main.py --ball_detector tracknet

# Chạy với YOLO26 PyTorch:
python main.py --ball_detector yolo  # hoặc dùng cờ ngắn: python main.py -b yolo

# Chạy với video khác:
python main.py --input "input_videos/new_input/clips/clip_02_zverev_vs_murray.mp4"

# Chạy live (bỏ qua cache stub để nhận diện lại từ đầu):
python main.py --no_stub

# Kết hợp chọn video, model bóng và chạy live:
python main.py -i "input_videos/new_input/clips/clip_01_nadal_vs_verdasco_fast.mp4" -b tracknet --no_stub
```

Video phân tích hoàn chỉnh sẽ được lưu tại thư mục `output_videos/`.

Video được đọc và xuất theo từng đoạn nhỏ để không phải giữ toàn bộ frame trong RAM. Dữ liệu nhận diện người, bóng và mốc sân được cache riêng; cache chỉ được dùng khi mới hơn video và trọng số model. Dùng `--no_stub` để nhận diện lại từ đầu.

Để kiểm tra YOLO có dùng GPU NVIDIA hay không trong môi trường `dat302m`:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Nếu kết quả là `False`, cài bản PyTorch CUDA phù hợp với GPU và driver theo [hướng dẫn chính thức](https://pytorch.org/get-started/locally/). `ball_batch_size` có thể giảm xuống `1` khi thiếu VRAM.

### 4. Công cụ cắt video clip từ trận đấu đầy đủ (`cut_video_clips.py`)
Khi bạn có một video trận đấu dài (Full Match) và muốn trích xuất các pha rally đẹp mắt, bạn chỉ cần dùng công cụ cắt clip cực kỳ tiện lợi:

**Cách 1: Chế độ hỏi đáp tương tác trực quan (Khuyến nghị ⭐)**
```powershell
python cut_video_clips.py
```
- Hệ thống sẽ tự động tìm video nguồn lớn nhất trong thư mục `input_videos/`.
- Nhập thời gian bắt đầu (dạng `Phút:Giây`, ví dụ `1:25` hoặc nhập số phút rồi enter số giây).
- Nhập thời gian kết thúc (dạng `Phút:Giây`, ví dụ `1:45` hoặc nhập số phút rồi enter số giây).
- Nhập tên file muốn đặt (ví dụ `clip_05_doubles`).
- Hệ thống sẽ cắt video trong vài giây, lưu vào `input_videos/new_input/clips/` và hỏi bạn có muốn tự động cấu hình vào `config.yaml` để chạy `python main.py` phân tích ngay lập tức hay không!

**Cách 2: Cắt nhanh bằng dòng lệnh một dòng (CLI)**
```powershell
# Dạng Phút:Giây
python cut_video_clips.py --start 01:25 --end 01:45 --name clip_doubles.mp4 --update_config

# Hoặc tách biệt rõ phút và giây:
python cut_video_clips.py --start_min 1 --start_sec 25 --end_min 1 --end_sec 45 --name clip_doubles.mp4
```

---

## 🏋️ Tập dữ liệu & Huấn luyện mô hình (Training & Datasets)

Hệ thống hỗ trợ huấn luyện lại các mô hình thành phần:
- **Tải Dataset đầy đủ**: [👉 Google Drive Datasets](https://drive.google.com/drive/folders/175Zhdm-b0HVc7F_x_Towv1SJtRBqv_GD?usp=sharing)
- **Tải Model Weights đã train**: [👉 Google Drive Model Weights](https://drive.google.com/drive/folders/1hjxjTtbpErMXYAl-4z5uH8cUV29_EchO)
- Xem chi tiết tổ chức dataset tại: [`DATASET_STRUCTURE.md`](DATASET_STRUCTURE.md) và tài liệu chi tiết quy trình tiền xử lý tại: [`DATA_PREPROCESSING.md`](DATA_PREPROCESSING.md).
- **Huấn luyện mô hình phát hiện bóng (YOLO26)**:
  ```powershell
  python training/train_yolo26_ball_detector.py --dataset merged --epochs 50 --batch 8
  ```
- **Huấn luyện mô hình vạch sân**: runtime hiện dùng kiến trúc/weights PyTorch của [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector/tree/e5cd4f1ce26b15361700d3d89e068cbf0e82749e). Script TensorFlow ResNet50 cũ trong `training/` chỉ được giữ làm tài liệu thử nghiệm và không tạo weights tương thích với runtime mới.

---

## 📂 Cấu trúc mã nguồn dự án

```text
tennis_analysis-main/
│
├── config.yaml                    # ⭐ File cấu hình trung tâm của toàn bộ hệ thống
├── main.py                        # ⭐ Script chạy phân tích chính (Pipeline Entrypoint)
├── cut_video_clips.py             # ✂️ Công cụ cắt clip nhanh theo Phút:Giây (Quick Clipper)
├── requirements.txt               # Danh sách thư viện Python phụ thuộc
├── .gitattributes                 # ⚙️ Cấu hình GitHub Linguist (nhận diện 100% Python, ẩn .ipynb)
├── .gitignore                     # Cấu hình bỏ qua models, datasets, cache và videos lớn
├── .env.example                   # Mẫu cấu hình biến môi trường
├── DATASET_STRUCTURE.md           # Tài liệu hướng dẫn cấu trúc dataset chi tiết
├── DATA_PREPROCESSING.md          # Tài liệu quy trình tiền xử lý dữ liệu chi tiết
├── README_STRUCTURE.md            # Tài liệu cấu trúc README dự án
│
├── models/                        # Thư mục chứa toàn bộ trọng số mô hình đã huấn luyện
│   ├── yolo26s.pt                 # Trọng số YOLO26 phát hiện người chơi
│   ├── ball_detector_yolo26_best.pt # Trọng số YOLO26 chuyên dụng phát hiện bóng
│   ├── tracknet_weights.pth       # Trọng số TrackNet PyTorch bám bóng chuẩn TenniSet
│   ├── model_tennis_court_det.pt  # TrackNet heatmap model phát hiện 14 điểm mốc sân
│   ├── bounce_model.cbm           # Mô hình CatBoost AI nhận diện điểm bóng nảy sân
│   └── ball_detector_tf_saved_model/# Mô hình bóng dạng TensorFlow SavedModel
│
├── court_line_detector/           # Module bám vạch sân (CPV Optical Flow)
│   ├── court_heatmap_model.py     # Kiến trúc TrackNet heatmap 15 channels
│   ├── court_postprocess.py       # Hough refinement + Otsu thresholding + homography reconstruction
│   ├── court_line_detector.py     # PyTorch detector khởi tạo 14 keypoints ở frame 0
│   ├── cpv_court_tracker.py       # ⭐ Multi-frame anchor screening + Bi-directional LK ngược về F0
│   └── court_segments.py          # Chọn các đoạn camera nhìn thấy đủ sân
│
├── trackers/                      # Module bám vết đối tượng
│   ├── player_tracker.py          # Player Tracker + ByteTrack + Đánh Đơn / Đánh Đôi
│   ├── ball_tracker.py            # ⭐ Ball Tracker Hybrid (TrackNet + YOLO) + Parabolic Lob Fitting
│   ├── tracknet_torch.py          # PyTorch TrackNet implementation & Sub-pixel Centroid
│   └── bounce_detector.py         # CatBoost bounce inference runtime
│
├── mini_court/                    # Module bản đồ sân 2D (Radar Mini-Court)
│   ├── mini_court.py              # Lớp điều phối MiniCourt trung tâm
│   ├── geometry.py                # Định nghĩa thông số kích thước & 28 keypoints sân
│   ├── projector.py               # Biến đổi Homography, 3D Ground Projection & Đồng bộ vượt lưới
│   └── drawer.py                  # Vẽ Radar 2D, lưới, sóng xung kích & chỉ số Caliper
│
├── utils/                         # Các hàm tiện ích bổ trợ
│   ├── referee_utils.py           # ⭐ Trọng tài VAR / Hawk-Eye ELC & Phán quyết điểm
│   ├── config_utils.py            # Trình đọc & hiển thị cấu hình YAML
│   ├── video_utils.py             # Đọc & ghi video OpenCV
│   ├── bbox_utils.py              # Xử lý hình học Bounding Box & khoảng cách
│   ├── conversions.py             # Chuyển đổi Pixel sang Mét
│   ├── player_stats_drawer_utils.py # Vẽ bảng thống kê tốc độ thi đấu
│   └── cut_video_clips.py         # Cắt clip từ video trận đấu đầy đủ
│
├── analysis/                      # ⭐ Module phân tích hậu trận đấu chuyên sâu (Post-Match Analytics)
│   ├── __init__.py                # Package exports
│   ├── court_geometry.py          # Hệ tọa độ & hình học sân tennis chuẩn ATP
│   ├── player_metrics.py          # Tính toán thể lực, quãng đường & 3 vùng sân cầu thủ
│   ├── ball_shot_metrics.py       # Phân tích vết bóng, điểm nảy Hawk-Eye & từng cú đánh
│   ├── data_exporter.py           # Quản lý xuất dữ liệu cấu trúc (JSON, CSV)
│   ├── visualizer.py              # Engine đồ họa vẽ bản đồ nhiệt 2D & biểu đồ (PNG)
│   └── match_analyzer.py          # Lớp điều phối chính (Facade Orchestrator) & CLI Runner
│
├── match_analytics/               # 📊 Thư mục chứa dữ liệu JSON/CSV và ảnh đồ họa sau trận
│   └── <tên_video>/               # Ví dụ: input_video_2/ (13 file dữ liệu & biểu đồ)
│
├── reports/                       # Báo cáo kỹ thuật và nghiên cứu tiền xử lý
│   ├── DATA_CLEANING_AND_PREPROCESSING.md # Báo cáo chi tiết các giai đoạn làm sạch dữ liệu
│   └── data_cleaning/             # Notebooks nghiên cứu tiền xử lý dữ liệu bóng và vạch sân
│
├── docs/                          # Tài liệu & Assets truyền thông
│   └── assets/                    # Video demo Full HD và ảnh GIF preview
│
├── input_videos/                  # Video đầu vào
│   └── new_input/clips/           # Các clip tình huống bóng mẫu (Nadal, Murray, Monfils,...)
│
├── output_videos/                 # Video kết quả sau khi phân tích
├── tracker_stubs/                 # Bộ đệm dữ liệu pickle (Cache detections tăng tốc độ)
└── training/                      # Mã nguồn và Notebooks phục vụ huấn luyện mô hình
```
