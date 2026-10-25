"""
Tennis Analysis - Trang giới thiệu (entrypoint)
Chạy:  streamlit run ui/app.py
Các trang con nằm trong ui/pages/:  1_setting.py (Cài đặt), 2_tennis_analysis.py (Phân tích), 3_dashboard.py (Dashboard)
Giao diện dùng chung hệ thiết kế với các trang còn lại.
"""
import streamlit as st
from streamlit.errors import StreamlitAPIException
from ui_common import render_sidebar

st.set_page_config(
    page_title="Tennis Analysis",
    page_icon="🎾",
    layout="wide",
    #initial_sidebar_state="collapsed",
)
render_sidebar()

ANALYSIS_PAGE = "pages/2_tennis_analysis.py"
SETTING_PAGE = "pages/3_setting.py"
DASHBOARD_PAGE = "pages/1_dashboard.py"

# =====================================================================================
# CSS (cùng tokens với Cài đặt / Phân tích / Dashboard)
# =====================================================================================
st.markdown(
    """
<style>
:root{--acc:#3b5bdb;--acc-2:#5c7cfa;--acc-soft:rgba(92,124,250,.16);--line:rgba(128,128,128,.26);
--surf:rgba(128,128,128,.07);--ok:#2fb344;--t:.18s ease}
header[data-testid="stHeader"]{background:transparent!important;pointer-events:none}
header[data-testid="stHeader"] *{pointer-events:auto}
.block-container{padding-top:4.2rem!important;padding-bottom:4rem}
/* ---------- Brand bar ---------- */
.brand{display:flex;align-items:center;gap:12px;margin-top:6px;margin-bottom:32px}
.brand .lg{width:42px;height:42px;border-radius:12px;display:grid;place-items:center;color:#fff;
background:linear-gradient(135deg,#182254,#3b5bdb);box-shadow:0 8px 20px rgba(59,91,219,.32)}
.brand b{font-size:17px;font-weight:750;letter-spacing:-.01em;color:var(--text-color)}
/* ---------- Hero ---------- */
.pill{display:inline-flex;align-items:center;gap:8px;padding:6px 13px;border-radius:999px;font-size:12.5px;font-weight:600;
border:1px solid rgba(92,124,250,.45);background:var(--acc-soft)}
.pill i{width:8px;height:8px;border-radius:50%;background:var(--ok);box-shadow:0 0 0 3px rgba(47,179,68,.22)}
.hero{font-weight:800;font-size:clamp(34px,4.6vw,54px);line-height:1.1;letter-spacing:-.025em;margin:20px 0 0!important;padding:0!important}
.hero .accent{display:block;background:linear-gradient(90deg,#5c7cfa 0%,#22d3ee 100%);-webkit-background-clip:text;background-clip:text;
color:transparent;-webkit-text-fill-color:transparent;padding-bottom:4px}
.lead{opacity:.75;font-size:16px;line-height:1.75;margin:18px 0 26px;max-width:520px}
/* ---------- Radar illustration ---------- */
.rad{border-radius:22px;padding:14px 14px 12px;color:#e8ecf8;border:1px solid rgba(128,140,180,.3);
background:radial-gradient(120% 90% at 20% 0%,#24336f 0%,#121829 62%);box-shadow:0 24px 50px rgba(20,30,80,.35)}
.rad-h{display:flex;justify-content:space-between;align-items:center;font-size:11.5px;font-weight:700;letter-spacing:.12em;margin:0 4px 10px}
.rad-h em{font-style:normal;color:#4ade80;display:flex;align-items:center;gap:6px}
.rad-h em:before{content:"";width:7px;height:7px;border-radius:50%;background:#4ade80;animation:blink 1.6s infinite}
.rad svg{display:block;width:100%;height:auto}
.rad-f{display:flex;flex-wrap:wrap;gap:8px 16px;font-size:12px;color:#aab3cf;margin:10px 4px 0}
.rad-f span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px;background:var(--c)}
.ring{transform-origin:232px 160px;animation:ring 2.2s ease-out infinite}
@keyframes ring{0%{transform:scale(.5);opacity:.9}100%{transform:scale(2.1);opacity:0}}
@keyframes blink{50%{opacity:.25}}
/* ---------- Stats ---------- */
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:44px 0 0}
.stat{border:1px solid var(--line);border-radius:16px;background:var(--surf);padding:18px 20px}
.stat .v{font-size:28px;font-weight:800;letter-spacing:-.02em;line-height:1.1}
.stat .l{font-size:13px;opacity:.7;margin-top:4px;line-height:1.45}
/* ---------- Sections ---------- */
.sh{margin:64px 0 22px}
.sh .k{font-size:12.5px;font-weight:700;letter-spacing:.14em;color:var(--acc-2);text-transform:uppercase}
.sh h2{margin:6px 0 0!important;padding:0!important;font-size:28px!important;font-weight:750!important;letter-spacing:-.015em;line-height:1.25!important}
.sh p{margin:8px 0 0;opacity:.7;font-size:15px;line-height:1.6;max-width:620px}
.fg{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:16px}
.fc{border:1px solid var(--line);border-radius:18px;background:var(--surf);padding:22px;transition:border-color var(--t),transform var(--t)}
.fc:hover{border-color:rgba(92,124,250,.65);transform:translateY(-2px)}
.fc .ic{width:44px;height:44px;border-radius:12px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2);margin-bottom:16px}
.fc b{display:block;font-size:16.5px;margin-bottom:6px}
.fc span{display:block;font-size:14px;opacity:.72;line-height:1.6}
/* ---------- Steps ---------- */
.sg{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;counter-reset:s}
.sp{position:relative;border:1px solid var(--line);border-radius:18px;padding:22px;background:var(--surf)}
.sp .n{width:34px;height:34px;border-radius:50%;display:grid;place-items:center;font-weight:800;font-size:15px;color:#fff;background:var(--acc);margin-bottom:14px}
.sp b{display:block;font-size:16px;margin-bottom:6px}
.sp span{display:block;font-size:13.5px;opacity:.72;line-height:1.6}
.sp:not(:last-child):after{content:"";position:absolute;top:39px;right:-17px;width:18px;height:2px;background:var(--line)}
@media (max-width:900px){.sp:after{display:none}}
/* ---------- Tech chips ---------- */
.tc{display:flex;flex-wrap:wrap;gap:10px}
.tc span{font-family:"Source Code Pro",ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;font-weight:600;padding:7px 13px;
border:1px solid var(--line);border-radius:8px;background:var(--surf)}
/* ---------- Buttons ---------- */
div.stButton>button{min-height:50px;border-radius:12px;font-weight:600;font-size:15px;transition:background var(--t),border-color var(--t),transform var(--t)}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{background:var(--acc);border:1px solid var(--acc);color:#fff;box-shadow:0 10px 24px rgba(59,91,219,.32)}
div.stButton>button[kind="primary"]:hover,div.stButton>button[data-testid="stBaseButton-primary"]:hover{background:#2f4bc4;border-color:#2f4bc4;color:#fff;transform:translateY(-1px)}
div.stButton>button:focus-visible{outline:2px solid var(--acc-2);outline-offset:2px}
/* ---------- CTA ---------- */
[class*="st-key-card_cta"]{border:0!important;border-radius:22px!important;padding:34px 36px!important;color:#fff;
background:linear-gradient(135deg,#182254 0%,#2a3f95 55%,#3a57c0 100%);box-shadow:0 18px 40px rgba(40,60,140,.3)}
[class*="st-key-card_cta"] h3{color:#fff!important;margin:0 0 6px!important;padding:0!important;font-size:24px!important;font-weight:750!important}
[class*="st-key-card_cta"] p{color:rgba(255,255,255,.78);margin:0 0 6px}
[class*="st-key-card_cta"] div.stButton>button{background:#fff;color:#182254;border:0}
[class*="st-key-card_cta"] div.stButton>button:hover{background:#e7ecff;color:#182254}
[class*="st-key-card_cta"] div.stButton>button[kind="secondary"],[class*="st-key-card_cta"] div.stButton>button[data-testid="stBaseButton-secondary"]{background:rgba(255,255,255,.12);color:#fff;border:1px solid rgba(255,255,255,.4)}
[class*="st-key-card_cta"] div.stButton>button[kind="secondary"]:hover{background:rgba(255,255,255,.2);color:#fff}
.foot{text-align:center;opacity:.55;font-size:12.5px;margin-top:34px}
@media (max-width:640px){.sh{margin-top:48px}[class*="st-key-card_cta"]{padding:26px 22px!important}}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation:none!important}.fc:hover{transform:none}}
</style>
""",
    unsafe_allow_html=True,
)

# =====================================================================================
# Icon SVG (cùng bộ nét 1.8px với các trang khác, không emoji)
# =====================================================================================
_ICONS = {
    "person": '<circle cx="12" cy="7" r="3.5"/><path d="M5 21c0-4 3.1-7 7-7s7 3 7 7"/>',
    "ball": '<circle cx="12" cy="12" r="9"/><path d="M5 6c4 3 4 9 0 12M19 6c-4 3-4 9 0 12"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "radar": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><path d="M12 12l6-6"/>',
    "gauge": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "layers": '<path d="M12 3l9 5-9 5-9-5 9-5z"/><path d="M3 13l9 5 9-5"/>',
    "download": '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>',
}


def svg(name, size=22):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def go(page):
    """Chuyển trang; báo lỗi rõ ràng nếu tên file trong ui/pages/ không khớp."""
    try:
        st.switch_page(page)
    except StreamlitAPIException:
        st.error(f"Không mở được trang `{page}`. Kiểm tra lại tên file trong thư mục `ui/pages/` "
                 "và các hằng số ANALYSIS_PAGE / SETTING_PAGE / DASHBOARD_PAGE ở đầu file này.")


# =====================================================================================
# Minh hoạ radar sân 2D (SVG tĩnh, hệ toạ độ 360x430)
# =====================================================================================
RADAR_SVG = (
    '<svg viewBox="0 0 360 430" role="img" aria-label="Minh hoạ radar sân tennis 2D với quỹ đạo bóng và điểm chạm IN">'
    '<rect x="0" y="0" width="360" height="430" rx="14" fill="#1b3f78"/>'
    '<rect x="50" y="40" width="260" height="350" fill="#2457a3"/>'
    '<g fill="none" stroke="#fff" stroke-width="2" stroke-linecap="round">'
    '<rect x="50" y="40" width="260" height="350"/><line x1="82" y1="40" x2="82" y2="390"/><line x1="278" y1="40" x2="278" y2="390"/>'
    '<line x1="82" y1="126" x2="278" y2="126"/><line x1="82" y1="304" x2="278" y2="304"/><line x1="180" y1="126" x2="180" y2="304"/></g>'
    '<line x1="40" y1="215" x2="320" y2="215" stroke="#ffb020" stroke-width="3.5" stroke-linecap="round"/>'
    '<text x="314" y="208" fill="#fff" font-size="10" font-weight="700" font-family="sans-serif" opacity=".9" text-anchor="end">NET</text>'
    '<path d="M140 350 Q205 250 232 160" fill="none" stroke="#d6ff3a" stroke-width="2.4" stroke-dasharray="2 7" stroke-linecap="round"/>'
    '<path d="M232 160 Q250 120 218 96" fill="none" stroke="rgba(214,255,58,.45)" stroke-width="2" stroke-dasharray="2 7" stroke-linecap="round"/>'
    '<circle class="ring" cx="232" cy="160" r="12" fill="none" stroke="#4ade80" stroke-width="2"/>'
    '<ellipse cx="232" cy="160" rx="9" ry="6" fill="#22b45a" stroke="#fff" stroke-width="1.6"/>'
    '<rect x="248" y="136" width="82" height="22" rx="11" fill="#121829" stroke="#22b45a" stroke-width="1.4"/>'
    '<text x="289" y="151" text-anchor="middle" fill="#fff" font-size="11" font-weight="700" font-family="sans-serif">IN +12cm</text>'
    '<circle cx="218" cy="96" r="9" fill="#ff6b4a" stroke="#fff" stroke-width="1.4"/>'
    '<text x="218" y="99.5" text-anchor="middle" fill="#fff" font-size="9" font-weight="700" font-family="sans-serif">P2</text>'
    '<circle cx="140" cy="352" r="9" fill="#4aa8ff" stroke="#fff" stroke-width="1.4"/>'
    '<text x="140" y="355.5" text-anchor="middle" fill="#fff" font-size="9" font-weight="700" font-family="sans-serif">P1</text>'
    '<circle cx="205" cy="241" r="5.5" fill="#d6ff3a"/>'
    '</svg>'
)

# =====================================================================================
# Brand + Hero
# =====================================================================================
st.markdown(
    f'<div class="brand"><div class="lg">{svg("ball", 22)}</div><b>Tennis Analysis</b></div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.12, 1], gap="large", vertical_alignment="center")
with left:
    st.markdown(
        '<span class="pill"><i></i>Xử lý cục bộ · Không cần internet</span>'
        '<h1 class="hero">Phân tích tennis thông minh'
        '<span class="accent">Hawk-Eye Radar &amp; Match Statistics</span></h1>'
        '<p class="lead">Từ video trận đấu đến radar sân 2D, thống kê tốc độ và phán quyết IN/OUT. '
        'Theo dõi cầu thủ và bóng, xuất video, CSV, JSON chỉ trong vài bước.</p>',
        unsafe_allow_html=True,
    )
    if st.button("Bắt đầu phân tích video", type="primary", width="stretch", key="hero_start"):
        go(ANALYSIS_PAGE)
    s1, s2 = st.columns(2)
    if s1.button("Dashboard", width="stretch", key="hero_dash"):
        go(DASHBOARD_PAGE)
    if s2.button("Cài đặt", width="stretch", key="hero_set"):
        go(SETTING_PAGE)
with right:
    st.markdown(
        '<div class="rad"><div class="rad-h"><span>HAWK-EYE 2D RADAR</span><em>LIVE</em></div>'
        f'{RADAR_SVG}'
        '<div class="rad-f"><span style="--c:#4aa8ff">Player 1</span><span style="--c:#ff6b4a">Player 2</span>'
        '<span style="--c:#d6ff3a">Quỹ đạo bóng</span><span style="--c:#22b45a">Điểm chạm</span></div></div>',
        unsafe_allow_html=True,
    )

# =====================================================================================
# Stats strip
# =====================================================================================
STATS = [
    ("14", "điểm mốc vạch sân được dò và bám theo từng khung hình"),
    ("2 – 4", "cầu thủ: tự phân biệt đánh đơn và đánh đôi"),
    ("IN / OUT", "phán quyết Hawk-Eye ELC kèm khoảng cách tới vạch"),
    ("Offline", "toàn bộ xử lý chạy trên máy của bạn"),
]
st.markdown(
    '<div class="stats">' + "".join(f'<div class="stat"><div class="v">{v}</div><div class="l">{l}</div></div>' for v, l in STATS) + "</div>",
    unsafe_allow_html=True,
)

# =====================================================================================
# Features
# =====================================================================================
FEATURES = [
    ("person", "Theo dõi cầu thủ", "YOLO26 nhận diện và bám cầu thủ, tự phân biệt đánh đơn và đánh đôi."),
    ("ball", "Theo dõi bóng", "TrackNet kết hợp YOLO (hybrid), xử lý tốt bóng nảy sân và motion blur."),
    ("target", "Dò vạch sân", "14 điểm mốc, bám từng pixel bằng Optical Flow và RANSAC."),
    ("radar", "Radar sân 2D", "Radar mini đồng bộ theo thời gian video, hiển thị vị trí người và bóng."),
    ("gauge", "Thống kê tốc độ", "Tốc độ cú đánh và tốc độ di chuyển của từng cầu thủ."),
    ("eye", "Trọng tài Hawk-Eye ELC", "Phán quyết IN/OUT, khoảng cách tới vạch và người ăn điểm."),
]
st.markdown(
    '<div class="sh"><div class="k">Tính năng</div><h2>Mọi thứ bạn cần để đọc một trận đấu</h2>'
    '<p>Sáu mô-đun phối hợp với nhau, từ nhận diện đến phán quyết, trong một giao diện duy nhất.</p></div>'
    '<div class="fg">' + "".join(
        f'<div class="fc"><div class="ic">{svg(ic)}</div><b>{t}</b><span>{d}</span></div>' for ic, t, d in FEATURES) + "</div>",
    unsafe_allow_html=True,
)

# =====================================================================================
# Steps
# =====================================================================================
STEPS = [
    ("Tải video", "Tải lên video trận đấu hoặc chọn clip có sẵn, chọn chế độ đánh đơn hoặc đánh đôi."),
    ("Phân tích", "Hệ thống theo dõi cầu thủ, bóng, vạch sân và phát hiện các điểm bóng chạm đất."),
    ("Xem kết quả", "Radar, bảng thống kê và thẻ Hawk-Eye đồng bộ theo từng khung hình của video."),
    ("Xuất và lưu", "Tải MP4, CSV, JSON, rồi lưu phiên vào lịch sử để theo dõi trên Dashboard."),
]
st.markdown(
    '<div class="sh"><div class="k">Quy trình</div><h2>Bốn bước từ video đến báo cáo</h2></div>'
    '<div class="sg">' + "".join(
        f'<div class="sp"><div class="n">{i}</div><b>{t}</b><span>{d}</span></div>' for i, (t, d) in enumerate(STEPS, 1)) + "</div>",
    unsafe_allow_html=True,
)

# =====================================================================================
# Tech stack
# =====================================================================================
CHIPS = ["yolo26", "tracknet", "optical flow", "homography", "hawk-eye elc", "catboost bounce"]
st.markdown(
    '<div class="sh"><div class="k">Công nghệ</div><h2>Xây dựng trên nền tảng thị giác máy tính</h2></div>'
    '<div class="tc">' + "".join(f"<span>{c}</span>" for c in CHIPS) + "</div>",
    unsafe_allow_html=True,
)

# =====================================================================================
# Final CTA
# =====================================================================================
st.write("")
st.write("")
try:
    cta = st.container(border=True, key="card_cta")
except TypeError:
    cta = st.container(border=True)
with cta:
    st.markdown("### Sẵn sàng phân tích trận đấu?\nTải video lên và nhận kết quả đầy đủ chỉ sau một lần xử lý.")
    c1, c2, c3 = st.columns([1.6, 1, 1])
    if c1.button("Bắt đầu phân tích video", type="primary", width="stretch", key="cta_start"):
        go(ANALYSIS_PAGE)
    if c2.button("Dashboard", width="stretch", key="cta_dash"):
        go(DASHBOARD_PAGE)
    if c3.button("Cài đặt", width="stretch", key="cta_set"):
        go(SETTING_PAGE)

st.markdown('<div class="foot">Tennis Analysis · YOLO26 · TrackNet · Optical Flow · Hawk-Eye ELC</div>', unsafe_allow_html=True)