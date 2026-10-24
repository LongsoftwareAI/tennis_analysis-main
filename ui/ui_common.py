"""
Bootstrap dùng chung cho các trang Streamlit trong ui/.
- Xác định ROOT (thư mục gốc dự án, cha của ui/), thêm vào sys.path, chdir về ROOT.
- Nạp utils/config_utils.py trực tiếp theo đường dẫn file (không import cả package `utils`,
  tránh kéo theo cv2/pandas/constants chỉ để đọc cấu hình).
"""
import importlib.util
import os
import sys
import streamlit as st

UI_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(UI_DIR)
CONFIG_PATH = os.path.join(ROOT, "config.yaml")

if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)


def _load_config_utils():
    path = os.path.join(ROOT, "utils", "config_utils.py")
    spec = importlib.util.spec_from_file_location("_tennis_config_utils", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cfgu = _load_config_utils()
_NAV_ITEMS = [
    ("app.py",                     "Trang chủ",       "home"),
    ("pages/1_dashboard.py",       "Dashboard",       "space_dashboard"),
    ("pages/2_tennis_analysis.py", "Phân tích Video", "play_circle"),
    ("pages/3_setting.py",         "Cài đặt",         "settings"),
]

def render_sidebar():
    with st.sidebar:
        # 1. Custom CSS để đồng bộ màu sắc sidebar với hệ thống Tennis
        st.markdown("""
        <style>
            /* Tự động thích ứng linh hoạt theo Theme Sáng / Tối (Streamlit Light / Dark Mode) */
            [data-testid="stSidebar"] {
                background-color: var(--secondary-background-color) !important;
                border-right: 1px solid rgba(128, 140, 180, 0.18) !important;
            }
            [data-testid="stSidebarNav"] {
                display: none; /* Ẩn menu mặc định của Streamlit */
            }
            .sidebar-brand {
                padding: 8px 0 16px 0;
            }
            .sidebar-brand .tag {
                color: #3b5bdb; font-size: 11px; font-weight: 800; letter-spacing: 1.5px;
            }
            .sidebar-brand .title {
                color: var(--text-color); font-size: 22px; font-weight: 800; margin: 6px 0; letter-spacing: -0.01em;
            }
            .sidebar-brand .sub {
                color: var(--text-color); opacity: 0.68; font-size: 11.5px; font-weight: 650; letter-spacing: 0.5px;
            }
            .sidebar-hr {
                margin: 0 0 18px 0; border: 0; border-top: 1px solid rgba(128, 140, 180, 0.2);
            }
            .runtime-box {
                background: rgba(128, 128, 128, 0.08);
                border: 1px solid rgba(128, 140, 180, 0.2);
                border-radius: 12px;
                padding: 16px;
                margin-top: 36px;
            }
            .runtime-header {
                display: flex; justify-content: space-between; align-items: center; margin-bottom: 14px;
            }
            .runtime-title {
                font-size: 11px; font-weight: 800; color: var(--text-color); opacity: 0.72; letter-spacing: 1.2px;
            }
            .runtime-badge {
                background: rgba(47, 179, 68, 0.15); color: #2fb344; border: 1px solid rgba(47, 179, 68, 0.3); padding: 3px 9px; border-radius: 6px; font-size: 11px; font-weight: 800;
            }
            .runtime-row {
                display: flex; justify-content: space-between; font-size: 13px; margin-bottom: 9px;
            }
            .runtime-row:last-child {
                margin-bottom: 0;
            }
            .runtime-label {
                color: var(--text-color); opacity: 0.72; font-weight: 500;
            }
            .runtime-value {
                color: var(--text-color); font-weight: 700;
            }
            [data-testid="stSidebar"] a {
                color: var(--text-color) !important;
                opacity: 0.85;
                border-radius: 10px !important;
                padding: 8px 12px !important;
                margin-bottom: 4px !important;
                border: 1px solid transparent !important;
                transition: all 0.16s ease !important;
            }
            [data-testid="stSidebar"] a p,
            [data-testid="stSidebar"] a span,
            [data-testid="stSidebar"] a svg,
            [data-testid="stSidebar"] a i {
                color: inherit !important;
            }

            /* Hiệu ứng khi di chuột qua menu */
            [data-testid="stSidebar"] a:hover {
                background-color: rgba(59, 91, 219, 0.08) !important;
                color: #3b5bdb !important;
                opacity: 1 !important;
                transform: translateX(2px);
            }

            /* Làm nổi bật trang đang chọn (Active page) */
            [data-testid="stSidebar"] a[aria-current="page"] {
                background-color: rgba(59, 91, 219, 0.14) !important;
                color: #3b5bdb !important;
                font-weight: 700 !important;
                opacity: 1 !important;
                border-color: rgba(59, 91, 219, 0.28) !important;
            }
        </style>
        """, unsafe_allow_html=True)

        # 2. Header: Tên phần mềm và Sub-title
        st.markdown("""
        <div class="sidebar-brand">
            <div class="tag">// TENNIS ASSISTANT</div>
            <div class="title">Trợ lý AI Thể thao</div>
            <div class="sub">TRACKING · HAWK-EYE · STATS</div>
        </div>
        <hr class="sidebar-hr">
        """, unsafe_allow_html=True)

        # 3. Navigation Menu (Sử dụng st.page_link để chuyển trang mượt mà)
        for path, label, icon in _NAV_ITEMS:
            st.page_link(path, label=label, icon=f":material/{icon}:")

        # 4. Footer: Box Runtime Status giả lập như trong ảnh
        st.markdown("""
        <div class="runtime-box">
            <div class="runtime-header">
                <span class="runtime-title">RUNTIME</span>
                <span class="runtime-badge">✓ OFFLINE</span>
            </div>
            <div class="runtime-row">
                <span class="runtime-label">Ball Engine</span>
                <span class="runtime-value">Hybrid (TrackNet)</span>
            </div>
            <div class="runtime-row">
                <span class="runtime-label">Court Engine</span>
                <span class="runtime-value">CPV Optical</span>
            </div>
        </div>
        """, unsafe_allow_html=True)