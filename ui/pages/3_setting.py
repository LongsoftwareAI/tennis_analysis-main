"""
Cài đặt - Tennis Analysis (giao diện hiện đại)
Logic lưu / khôi phục giữ nguyên: cfgu.load_config / save_user_config / reset_user_config.
Thay đổi chính: card theo nhóm, toggle thay checkbox, segmented control, icon SVG (không emoji),
trạng thái "chưa lưu", thanh hành động rõ ràng (1 nút chính duy nhất).
"""
import html
import os
import sys

import streamlit as st

# ui/ (chứa ui_common.py) phải nằm trong sys.path dù chạy từ đâu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_common import ROOT, CONFIG_PATH, cfgu,render_sidebar  # noqa: E402  (cũng thêm ROOT vào sys.path)

st.set_page_config(page_title="Cài đặt - Tennis Analysis", page_icon="⚙️", layout="wide")
render_sidebar()

ss = st.session_state
cfg = cfgu.load_config(CONFIG_PATH)
trk, mdl, vis, ply = cfg["tracking"], cfg["models"], cfg["visualization"], cfg["players"]
user_file = cfgu.user_override_path(CONFIG_PATH)

# =====================================================================================
# Design tokens + CSS  (màu theo rgba để hợp cả theme sáng lẫn tối)
# =====================================================================================
st.markdown("""
<style>
:root{--acc:#3b5bdb;--acc-2:#5c7cfa;--acc-soft:rgba(92,124,250,.16);--line:rgba(128,128,128,.26);
--surf:rgba(128,128,128,.07);--ok:#2fb344;--ok-soft:rgba(47,179,68,.15);--warn:#f59f00;--warn-soft:rgba(245,159,0,.16);--t:.16s ease}
header[data-testid="stHeader"]{background:transparent!important;pointer-events:none}
header[data-testid="stHeader"] *{pointer-events:auto}
.block-container{padding-top:4.2rem!important;padding-bottom:4rem}
/* ---------- Header ---------- */
.pg-hd{display:flex;gap:16px;align-items:center;margin-top:6px}
.pg-ic{flex:none;width:52px;height:52px;border-radius:15px;display:grid;place-items:center;color:#fff;
background:linear-gradient(135deg,#182254 0%,#3b5bdb 100%);box-shadow:0 10px 24px rgba(59,91,219,.32)}
.pg-t{margin:0!important;padding:0!important;font-size:28px!important;font-weight:750!important;line-height:1.2!important;letter-spacing:-.01em;color:var(--text-color)}
.pg-s{margin:4px 0 0;opacity:.72;font-size:14px;line-height:1.55;color:var(--text-color)}
.pg-s code{font-size:12.5px;padding:2px 6px;border-radius:6px;background:var(--surf)}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 6px}
/* ---------- Chips ---------- */
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font-size:12.5px;font-weight:600;
line-height:1.2;border:1px solid var(--line);background:var(--surf);white-space:nowrap}
.chip.ok{background:var(--ok-soft);border-color:rgba(47,179,68,.45)}
.chip.ok svg{color:var(--ok)}
.chip.warn{background:var(--warn-soft);border-color:rgba(245,159,0,.5)}
.chip.warn svg{color:var(--warn)}
.chip.acc{background:var(--acc-soft);border-color:rgba(92,124,250,.45)}
.chip.acc svg{color:var(--acc-2)}
.chip svg{flex:none}
.stat-r{display:flex;justify-content:flex-end;align-items:center;height:52px}
/* ---------- Tabs (pill) ---------- */
.stTabs [data-baseweb="tab-list"]{gap:4px;background:var(--surf);padding:4px;border-radius:13px;width:fit-content;max-width:100%;border:1px solid var(--line)}
.stTabs [data-baseweb="tab-highlight"],.stTabs [data-baseweb="tab-border"]{display:none}
.stTabs [data-baseweb="tab"]{height:40px;padding:0 20px;border-radius:10px;font-weight:600;background:transparent;transition:background var(--t),color var(--t)}
.stTabs [data-baseweb="tab"][aria-selected="true"]{background:var(--acc);color:#fff}
.stTabs [data-baseweb="tab"][aria-selected="true"] p{color:#fff}
.stTabs [data-baseweb="tab"]:focus-visible{outline:2px solid var(--acc-2);outline-offset:2px}
.stTabs [data-baseweb="tab-panel"]{padding-top:22px}
/* ---------- Cards ---------- */
[class*="st-key-card_"]{border:1px solid var(--line)!important;border-radius:18px!important;background:var(--surf);padding:22px 24px!important}
.sec{display:flex;gap:12px;align-items:flex-start;margin:0 0 16px}
.sec-ic{flex:none;width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2)}
.sec-t{font-size:16px;font-weight:700;line-height:1.3}
.sec-d{font-size:13px;opacity:.7;line-height:1.5;margin-top:2px}
.hint{font-size:13px;opacity:.72;line-height:1.5;margin:-4px 0 12px}
.row-d{font-size:12.5px;opacity:.68;line-height:1.45;margin:-6px 0 14px 0}
/* ---------- Inputs ---------- */
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{border-radius:10px!important}
[data-testid="stTextInput"] input,[data-testid="stNumberInput"] input{min-height:44px}
[data-testid="stWidgetLabel"] p{font-weight:600;font-size:13.5px}
/* ---------- Segmented control (st.radio ngang) ---------- */
[data-testid="stRadio"] div[role="radiogroup"]{gap:4px;background:var(--surf);padding:4px;border-radius:12px;width:fit-content;max-width:100%;flex-wrap:wrap;border:1px solid var(--line)}
[data-testid="stRadio"] label{margin:0!important;padding:8px 16px;border-radius:9px;cursor:pointer;transition:background var(--t),color var(--t)}
[data-testid="stRadio"] label>div:first-child{display:none}
[data-testid="stRadio"] label:hover{background:var(--acc-soft)}
[data-testid="stRadio"] label:has(input:checked){background:var(--acc);color:#fff}
[data-testid="stRadio"] label:has(input:checked) *{color:#fff!important}
[data-testid="stRadio"] label:has(input:focus-visible){outline:2px solid var(--acc-2);outline-offset:2px}
[data-testid="stRadio"] label:has(input:disabled){opacity:.45;cursor:not-allowed}
/* ---------- Buttons ---------- */
div.stButton>button{min-height:46px;border-radius:12px;font-weight:600;transition:background var(--t),border-color var(--t),transform var(--t)}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{background:var(--acc);border:1px solid var(--acc);color:#fff}
div.stButton>button[kind="primary"]:hover,div.stButton>button[data-testid="stBaseButton-primary"]:hover{background:#2f4bc4;border-color:#2f4bc4;color:#fff}
div.stButton>button:focus-visible{outline:2px solid var(--acc-2);outline-offset:2px}
div.stButton>button:disabled{opacity:.42}
.bar-st{display:flex;align-items:center;gap:10px;min-height:46px;font-size:14px}
.bar-st b{font-weight:650}
.bar-st .sub{opacity:.68;font-size:12.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
""", unsafe_allow_html=True)

# =====================================================================================
# Icon SVG (một bộ nét 1.8px, không dùng emoji)
# =====================================================================================
_ICONS = {
    "sliders": '<line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/>'
               '<circle cx="9" cy="6" r="2" fill="currentColor"/><circle cx="15" cy="12" r="2" fill="currentColor"/><circle cx="7" cy="18" r="2" fill="currentColor"/>',
    "court": '<rect x="3" y="5" width="18" height="14" rx="2"/><line x1="12" y1="5" x2="12" y2="19"/><line x1="3" y1="12" x2="21" y2="12"/>',
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.5"/><path d="M17 14c2.5 0 4 2 4 5"/>',
    "person": '<circle cx="12" cy="7" r="3.5"/><path d="M5 21c0-4 3.1-7 7-7s7 3 7 7"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5"/>'
           '<path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "zap": '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
    "ball": '<circle cx="12" cy="12" r="9"/><path d="M5 6c4 3 4 9 0 12M19 6c-4 3-4 9 0 12"/>',
    "box": '<path d="M12 2l9 5v10l-9 5-9-5V7l9-5z"/><path d="M3 7l9 5 9-5M12 12v10"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/>',
    "alert": '<path d="M12 3l10 18H2L12 3z"/><line x1="12" y1="10" x2="12" y2="15"/><line x1="12" y1="18" x2="12" y2="18.1"/>',
    "dot": '<circle cx="12" cy="12" r="4" fill="currentColor"/>',
}


def svg(name, size=18):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def chip(text, kind="", icon=None):
    ic = svg(icon, 14) if icon else ""
    return f'<span class="chip {kind}">{ic}{html.escape(str(text))}</span>'


def sec(icon, title, desc=""):
    st.markdown(
        f'<div class="sec"><div class="sec-ic">{svg(icon, 18)}</div><div>'
        f'<div class="sec-t" role="heading" aria-level="2">{title}</div>'
        f'<div class="sec-d">{desc}</div></div></div>', unsafe_allow_html=True)


def card(name):
    """Card có viền; dùng key để CSS nhắm đúng (fallback nếu bản Streamlit cũ không hỗ trợ key)."""
    try:
        return st.container(border=True, key=f"card_{name}")
    except TypeError:
        return st.container(border=True)


# =====================================================================================
# Helper widget (mọi widget có key -> reset được sạch)
# =====================================================================================
KEY = lambda n: f"cfg_{n}"  # noqa: E731


def select(label, name, options, current, labels=None, help=None):
    options = list(options)
    if current not in options:
        options = [current] + options
    fmt = (lambda v: labels.get(v, v)) if labels else (lambda v: v)
    return st.selectbox(label, options, index=options.index(current), format_func=fmt, help=help, key=KEY(name))


def segmented(label, name, options, current, labels, disabled=False, help=None):
    options = list(options)
    if current not in options:
        options = [current] + options
    return st.radio(label, options, index=options.index(current), format_func=lambda v: labels.get(v, v),
                    horizontal=True, key=KEY(name), disabled=disabled, help=help)


def switch(label, name, value, desc="", disabled=False):
    v = st.toggle(label, value=bool(value), key=KEY(name), disabled=disabled)
    if desc:
        st.markdown(f'<div class="row-d">{desc}</div>', unsafe_allow_html=True)
    return v


MODEL_STATE = []   # (name, found) để đếm file model


def model_input(label, key_name, help=None):
    path = st.text_input(label, value=mdl.get(key_name, ""), help=help, key=KEY(f"mdl_{key_name}"))
    if path:
        full = path if os.path.isabs(path) else os.path.join(ROOT, path)
        ok = os.path.exists(full)
        MODEL_STATE.append(ok)
        st.markdown(chip("Đã tìm thấy file", "ok", "check") if ok
                    else chip("Không tìm thấy file - kiểm tra đường dẫn", "warn", "alert"), unsafe_allow_html=True)
    else:
        MODEL_STATE.append(False)
        st.markdown(chip("Chưa nhập đường dẫn", "warn", "alert"), unsafe_allow_html=True)
    return path


# =====================================================================================
# Flash message sau khi lưu / khôi phục
# =====================================================================================
if ss.get("_flash"):
    st.toast(ss.pop("_flash"))

# =====================================================================================
# Header
# =====================================================================================
h_l, h_r = st.columns([4, 1.4], vertical_alignment="center")
with h_l:
    st.markdown(
        f'<div class="pg-hd"><div class="pg-ic">{svg("sliders", 26)}</div><div>'
        f'<h1 class="pg-t">Cài đặt</h1>'
        f'<p class="pg-s">Lưu vào <code>{html.escape(os.path.basename(user_file))}</code>, ghi đè lên '
        f'<code>config.yaml</code> (comment trong <code>config.yaml</code> không bị thay đổi). '
        f'Áp dụng cho lần xử lý tiếp theo.</p></div></div>', unsafe_allow_html=True)
status_ph = h_r.empty()
meta_ph = st.empty()

# =====================================================================================
# Tabs
# =====================================================================================
tab_basic, tab_models, tab_vis = st.tabs(["Cơ bản", "Đường dẫn model", "Hiển thị"])

MODE_LABELS = {"auto": "Tự động", "singles": "Đánh đơn", "doubles": "Đánh đôi"}
MODE_HINT = {
    "auto": "Tự nhận diện đánh đơn (2 người) hoặc đánh đôi (4 người) từ video.",
    "singles": "Cố định đánh đơn: theo dõi 2 cầu thủ (P1 gần camera, P2 xa camera).",
    "doubles": "Cố định đánh đôi: theo dõi 4 cầu thủ (đội 1: P1 + P3, đội 2: P2 + P4).",
}

# ---------------------------------------------------------------- Cơ bản
with tab_basic:
    # 1. Card Bộ nhớ đệm (trải dài full width)
    with card("cache"):
        sec("zap", "Bộ nhớ đệm", "Tái sử dụng kết quả phát hiện để chạy nhanh hơn.")
        use_stubs = switch("Dùng cache stub nếu có", "use_stubs", trk["use_stubs"],
                           "Chỉ bật khi cùng tên video và nội dung video không đổi.")
        
    # 2. Card Tracking (trải dài full width, chia 2 cột bên trong)
    with card("tracking"):
        sec("cpu", "Tracking", "Thiết bị xử lý và các mô hình phát hiện.")
        
        t_l, t_r = st.columns(2, gap="large") # Chia 2 cột bên trong thẻ Tracking
        
        with t_l:
            device = select("Thiết bị xử lý (device)", "device", ["auto", "cuda", "cpu"], trk["device"],
                            labels={"auto": "Tự động - ưu tiên GPU CUDA, không có thì CPU",
                                    "cuda": "CUDA - bắt buộc GPU NVIDIA",
                                    "cpu": "CPU - bắt buộc dùng CPU"})
            court_mode = select("Chế độ bám vạch sân (court_mode)", "court_mode", ["cpv", "dynamic", "static"],
                                trk["court_mode"],
                                labels={"cpv": "cpv - Lucas-Kanade + RANSAC (khuyến nghị)",
                                        "dynamic": "dynamic", "static": "static"})
            
        with t_r:
            ball_detector = select("Mô hình phát hiện bóng (ball_detector)", "ball_detector",
                                   ["hybrid", "tracknet", "yolo"], trk["ball_detector"],
                                   labels={"hybrid": "hybrid - TrackNet + YOLO (khuyến nghị)",
                                           "tracknet": "tracknet - TrackNet 3 khung hình",
                                           "yolo": "yolo - YOLO26 đơn khung hình"})
            ball_batch_size = st.number_input("Ball batch size (ball_batch_size)", min_value=1, max_value=64,
                                              value=int(trk.get("ball_batch_size", 4)), step=1,
                                              key=KEY("batch"),
                                              help="Số khung hình bóng suy luận mỗi lượt. Lớn hơn = nhanh hơn nhưng tốn VRAM.")

# ---------------------------------------------------------------- Models
with tab_models:
    mdl_sum_ph = st.empty()
    ml, mr = st.columns(2, gap="large")
    with ml:
        with card("m_scene"):
            sec("person", "Người chơi & sân", "Phát hiện cầu thủ, 14 điểm mốc vạch sân và điểm bóng chạm đất.")
            m_player = model_input("Model người chơi (player_model)", "player_model")
            m_court = model_input("Vạch sân - 14 điểm mốc (court_model)", "court_model")
            m_bounce = model_input("Điểm bóng chạm đất - CatBoost (bounce_model)", "bounce_model")
    with mr:
        with card("m_ball"):
            sec("ball", "Phát hiện bóng", "Các mô hình bóng; model mặc định được chọn theo ball_detector.")
            m_tracknet = model_input("TrackNet (ball_model_tracknet)", "ball_model_tracknet",
                                     help="TrackNet PyTorch chuẩn TenniSet, tốt cho bóng nảy sân.")
            m_yolo = model_input("YOLO26 (ball_model_yolo)", "ball_model_yolo")
            m_ball = model_input("Bóng mặc định (ball_model)", "ball_model",
                                 help="Model bóng mặc định, tự kích hoạt theo ball_detector.")
    found = sum(MODEL_STATE)
    total = len(MODEL_STATE)
    mdl_sum_ph.markdown(
        '<div class="meta" style="margin-top:0">' +
        (chip(f"{found}/{total} file model hợp lệ", "ok", "check") if found == total
         else chip(f"{found}/{total} file model hợp lệ - còn {total - found} file cần kiểm tra", "warn", "alert")) +
        '</div>', unsafe_allow_html=True)

# ---------------------------------------------------------------- Hiển thị
with tab_vis:
    vl, vr = st.columns(2, gap="large")
    with vl:
        with card("v_obj"):
            sec("eye", "Đối tượng trên video", "Những gì được vẽ trực tiếp lên video đầu ra.")
            draw_players = switch("Vẽ người chơi", "draw_players", vis["draw_players"])
            player_draw_mode = segmented("Kiểu vẽ người chơi", "player_draw_mode", ["ellipse", "box"],
                                         vis["player_draw_mode"],
                                         {"ellipse": "Elip dưới chân", "box": "Khung chữ nhật"},
                                         disabled=not draw_players)
            st.write("")
            draw_ball = switch("Vẽ bóng tennis", "draw_ball", vis["draw_ball"])
            ball_effect = segmented("Hiệu ứng bóng", "ball_effect", ["tracer", "box"], vis["ball_effect"],
                                    {"tracer": "Đuôi sao băng", "box": "Khung chữ nhật"},
                                    disabled=not draw_ball)
            st.write("")
            draw_court_keypoints = switch("Vẽ 14 điểm mốc vạch sân", "draw_court_keypoints",
                                          vis["draw_court_keypoints"],
                                          "Hiển thị các điểm mốc vạch sân đã dò được lên video.")
    with vr:
        with card("v_over"):
            sec("layers", "Lớp phủ thông tin", "Radar và thống kê hiển thị cạnh video.")
            draw_mini_court = switch("Radar sân mini 2D", "draw_mini_court", vis["draw_mini_court"],
                                     "Bản đồ sân 2D đồng bộ theo thời gian video.")
            draw_stats = switch("Bảng thống kê tốc độ", "draw_stats", vis["draw_stats"],
                                "Tốc độ cú đánh và tốc độ di chuyển của từng cầu thủ.")
            draw_referee = switch("Thẻ phán quyết Hawk-Eye ELC", "draw_referee_decision",
                                  vis["draw_referee_decision"], "Kết quả IN/OUT, vạch gần nhất, người ăn điểm.")
            draw_frame_number = switch("Số thứ tự frame", "draw_frame_number", vis["draw_frame_number"],
                                       "Hữu ích khi gỡ lỗi hoặc đối chiếu dữ liệu theo frame.")

# =====================================================================================
# So sánh với cấu hình đang lưu -> trạng thái "chưa lưu"
# =====================================================================================
new_settings = {
    # Đã xóa khóa "players"
    "tracking": {
        "device": device,
        "court_mode": court_mode,
        "ball_detector": ball_detector,
        "ball_batch_size": int(ball_batch_size),
        # Đã xóa "match_mode": match_mode
        "use_stubs": use_stubs,
        "force_live": not use_stubs,
    },
    "models": {
        "player_model": m_player,
        "ball_model_tracknet": m_tracknet,
        "ball_model_yolo": m_yolo,
        "ball_model": m_ball,
        "court_model": m_court,
        "bounce_model": m_bounce,
    },
    "visualization": {
        "draw_players": draw_players,
        "player_draw_mode": player_draw_mode,
        "draw_ball": draw_ball,
        "ball_effect": ball_effect,
        "draw_court_keypoints": draw_court_keypoints,
        "draw_mini_court": draw_mini_court,
        "draw_stats": draw_stats,
        "draw_referee_decision": draw_referee,
        "draw_frame_number": draw_frame_number,
    },
}

baseline = {
    # Đã xóa khóa "players"
    "tracking": {
        "device": trk["device"], "court_mode": trk["court_mode"], "ball_detector": trk["ball_detector"],
        "ball_batch_size": int(trk.get("ball_batch_size", 4)), 
        # Đã xóa "match_mode": trk["match_mode"]
        "use_stubs": bool(trk["use_stubs"]), "force_live": not bool(trk["use_stubs"]),
    },
    "models": {k: mdl.get(k, "") for k in new_settings["models"]},
    "visualization": {
        "draw_players": bool(vis["draw_players"]), "player_draw_mode": vis["player_draw_mode"],
        "draw_ball": bool(vis["draw_ball"]), "ball_effect": vis["ball_effect"],
        "draw_court_keypoints": bool(vis["draw_court_keypoints"]),
        "draw_mini_court": bool(vis["draw_mini_court"]), "draw_stats": bool(vis["draw_stats"]),
        "draw_referee_decision": bool(vis["draw_referee_decision"]),
        "draw_frame_number": bool(vis["draw_frame_number"]),
    },
}
dirty = new_settings != baseline

# Header: trạng thái + tóm tắt cấu hình hiện tại (live)
status_ph.markdown(
    '<div class="stat-r">' + (chip("Có thay đổi chưa lưu", "warn", "dot") if dirty
                              else chip("Đã đồng bộ", "ok", "check")) + '</div>', unsafe_allow_html=True)
meta_ph.markdown(
    '<div class="meta">' +
    # Đã xóa dòng chip match_mode
    chip(f"Thiết bị: {device.upper()}", "acc", "cpu") +
    chip(f"Bóng: {ball_detector}", "acc", "ball") +
    chip("Cache: bật" if use_stubs else "Cache: tắt", "acc", "zap") +
    '</div>', unsafe_allow_html=True)

# =====================================================================================
# Thanh hành động (1 nút chính duy nhất)
# =====================================================================================
st.write("")
with card("actions"):
    a_l, a_r1, a_r2 = st.columns([5, 1.7, 1.7], vertical_alignment="center")
    with a_l:
        if dirty:
            st.markdown(f'<div class="bar-st">{chip("Chưa lưu", "warn", "dot")}'
                        f'<span class="sub">Nhấn "Lưu cài đặt" để áp dụng cho lần xử lý tiếp theo.</span></div>',
                        unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bar-st">{chip("Đã đồng bộ", "ok", "check")}'
                        f'<span class="sub">Cài đặt hiện tại trùng với cấu hình đang lưu.</span></div>',
                        unsafe_allow_html=True)
    do_reset = a_r1.button("Khôi phục mặc định", width='stretch', disabled=not os.path.exists(user_file),
                           help="Xoá cài đặt người dùng, quay về giá trị trong config.yaml.")
    do_save = a_r2.button("Lưu cài đặt", type="primary", width='stretch', disabled=not dirty)

if do_save:
    saved = cfgu.save_user_config(new_settings, CONFIG_PATH)
    ss["_flash"] = f"Đã lưu cài đặt vào {os.path.basename(str(saved))}"
    st.rerun()

if do_reset:
    if cfgu.reset_user_config(CONFIG_PATH):
        for k in [k for k in ss.keys() if str(k).startswith("cfg_")]:   # xoá state widget -> nạp lại mặc định
            del ss[k]
        ss["_flash"] = "Đã khôi phục về giá trị trong config.yaml"
        st.rerun()
    else:
        st.info("Chưa có cài đặt người dùng nào để khôi phục.")