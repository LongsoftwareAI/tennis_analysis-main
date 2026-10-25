"""
Tennis Analysis – Streamlit UI
Chạy:  streamlit run ui/app.py   (trang nằm trong ui/pages/)

Cách hoạt động: app KHÔNG import pipeline, mà chạy `main.py` như tiến trình con với một config tạm
(tắt radar / bảng thống kê / thẻ trọng tài / số frame trên video để có VIDEO SẠCH, bật analytics),
sau đó đọc các file CSV/JSON do analysis/match_analyzer.py (MatchAnalyzer) xuất ra
<TMP_DIR>/analytics/<video_stem>/ và vẽ lại radar + bảng thống kê đồng bộ theo thời gian video
bằng một component HTML.
"""
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from copy import deepcopy
from datetime import datetime
import html as _html

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import yaml

st.set_page_config(page_title="Tennis Analysis", page_icon="🎾", layout="wide")

# ui/ (chứa ui_common.py) phải nằm trong sys.path dù chạy từ đâu
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_common import ROOT, UI_DIR, CONFIG_PATH, cfgu, render_sidebar  # noqa: E402  (thêm ROOT vào sys.path + chdir ROOT)
render_sidebar()
UPLOAD_DIR = "uploads"
# Streamlit chỉ phục vụ thư mục static/ nằm cạnh file entrypoint (ui/app.py) -> ui/static/outputs
OUT_DIR = os.path.join(UI_DIR, "static", "outputs")   # phục vụ ở /app/static/outputs (cần enableStaticServing)
TMP_DIR = ".app_tmp"
ANALYTICS_ROOT = os.path.abspath(os.path.join(TMP_DIR, "analytics"))   # MatchAnalyzer ghi vào đây/<video_stem>/
CLIPS_GLOB = "input_videos/*.mp4"
HISTORY_DIR = os.path.join(ROOT, "history")
HISTORY_PATH = os.path.join(HISTORY_DIR, "sessions.csv")   # mỗi dòng = 1 phiên; trang Dashboard đọc file này
HISTORY_COLS = ["session_id", "saved_at", "video_name", "duration_sec",
                "max_shot_kmh", "distance_m", "hits",
                "decision", "point_winner", "nearest_line", "margin_cm", "landing_frame"]
for d in (UPLOAD_DIR, OUT_DIR, TMP_DIR, ANALYTICS_ROOT, HISTORY_DIR):
    os.makedirs(d, exist_ok=True)

# =====================================================================================
# Design tokens + CSS  (dùng chung với trang Cài đặt: rgba để hợp cả theme sáng/tối)
# =====================================================================================
st.markdown("""
<style>
:root{--acc:#3b5bdb;--acc-2:#5c7cfa;--acc-soft:rgba(92,124,250,.16);--line:rgba(128,128,128,.26);
--surf:rgba(128,128,128,.07);--ok:#2fb344;--ok-soft:rgba(47,179,68,.15);--bad:#e5484d;--bad-soft:rgba(229,72,77,.15);
--warn:#f59f00;--warn-soft:rgba(245,159,0,.16);--t:.16s ease}
header[data-testid="stHeader"]{background:transparent!important;pointer-events:none}
header[data-testid="stHeader"] *{pointer-events:auto}
.block-container{padding-top:4.2rem!important;padding-bottom:4rem}
/* Header */
.pg-hd{display:flex;gap:16px;align-items:center;margin-top:6px}
.pg-ic{flex:none;width:52px;height:52px;border-radius:15px;display:grid;place-items:center;color:#fff;
background:linear-gradient(135deg,#182254 0%,#3b5bdb 100%);box-shadow:0 10px 24px rgba(59,91,219,.32)}
.pg-t{margin:0!important;padding:0!important;font-size:28px!important;font-weight:750!important;line-height:1.2!important;letter-spacing:-.01em;color:var(--text-color)}
.pg-s{margin:4px 0 0;opacity:.72;font-size:14px;line-height:1.55;color:var(--text-color)}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 18px}
.nav-r{display:flex;justify-content:flex-end;align-items:center;height:52px}
/* Chips */
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font-size:12.5px;font-weight:600;
line-height:1.2;border:1px solid var(--line);background:var(--surf);white-space:nowrap}
.chip.ok{background:var(--ok-soft);border-color:rgba(47,179,68,.45)}.chip.ok svg{color:var(--ok)}
.chip.bad{background:var(--bad-soft);border-color:rgba(229,72,77,.5)}.chip.bad svg{color:var(--bad)}
.chip.warn{background:var(--warn-soft);border-color:rgba(245,159,0,.5)}.chip.warn svg{color:var(--warn)}
.chip.acc{background:var(--acc-soft);border-color:rgba(92,124,250,.45)}.chip.acc svg{color:var(--acc-2)}
.chip svg{flex:none}
/* Stepper */
.stepper{display:flex;gap:10px;align-items:center;margin:0 0 20px;flex-wrap:wrap}
.stp{display:flex;align-items:center;gap:10px;padding:7px 16px 7px 8px;border-radius:999px;font-weight:600;font-size:14px;
border:1px solid var(--line);background:var(--surf);opacity:.62}
.stp i{width:24px;height:24px;border-radius:50%;border:2px solid var(--line);display:grid;place-items:center;font-style:normal;font-size:12px}
.stp.active{opacity:1;border-color:rgba(92,124,250,.55);background:var(--acc-soft)}
.stp.active i{border-color:var(--acc-2);border-top-color:transparent;animation:sp 1s linear infinite;color:transparent}
.stp.done{opacity:1}.stp.done i{background:var(--ok);border-color:var(--ok);color:#fff}
.stp.done i svg{display:block}
.sbar{flex:1;height:2px;background:var(--line);min-width:20px;border-radius:2px}
@keyframes sp{to{transform:rotate(360deg)}}
/* Card / section */
[class*="st-key-card_"]{border:1px solid var(--line)!important;border-radius:18px!important;background:var(--surf);padding:22px 24px!important}
.sec{display:flex;gap:12px;align-items:flex-start;margin:0 0 16px}
.sec-ic{flex:none;width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2)}
.sec-t{font-size:16px;font-weight:700;line-height:1.3}.sec-d{font-size:13px;opacity:.7;line-height:1.5;margin-top:2px}
.hint{font-size:13px;opacity:.72;line-height:1.5;margin:-4px 0 12px}
.blk-t{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;margin:30px 0 14px}
.blk-t svg{color:var(--acc-2)}
/* KPI */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:14px;margin:0 0 18px}
.kpi{border:1px solid var(--line);border-radius:16px;background:var(--surf);padding:16px 18px;display:flex;gap:14px;align-items:center}
.kpi .ic{flex:none;width:40px;height:40px;border-radius:12px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2)}
.kpi .v{font-size:24px;font-weight:750;line-height:1.1;font-variant-numeric:tabular-nums}
.kpi .v small{font-size:13px;font-weight:600;opacity:.65;margin-left:4px}
.kpi .l{font-size:12.5px;opacity:.7;margin-top:3px}
/* Referee decision */
.dec{border-radius:16px;padding:16px 20px;background:var(--surf);display:flex;gap:22px;align-items:center;border:1px solid rgba(47,179,68,.55);
border-left:6px solid var(--ok);flex-wrap:wrap;margin-top:14px}
.dec.out{border-color:rgba(229,72,77,.55);border-left-color:var(--bad)}
.dec .badge{font-size:26px;font-weight:800;padding:8px 22px;border-radius:12px;background:var(--ok);color:#fff;letter-spacing:.02em}
.dec.out .badge{background:var(--bad)}
.dec dl{margin:0;display:grid;grid-template-columns:auto 1fr;gap:4px 18px;font-size:14px}
.dec dt{opacity:.68}.dec dd{margin:0;font-weight:650}
/* Gallery */
.ag{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:16px;margin:6px 0 4px}
.ac{border:1px solid var(--line);background:var(--surf);border-radius:18px;padding:8px 8px 12px;transition:border-color var(--t),transform var(--t)}
.ac:hover{border-color:rgba(92,124,250,.6)}
.ac .th{display:block;width:100%;aspect-ratio:4/3;border-radius:12px;overflow:hidden;background:#0b0f19}
.ac .th img{width:100%;height:100%;object-fit:cover;object-position:center;display:block;transition:transform .25s}
.ac:hover .th img{transform:scale(1.04)}
.ac .ft{display:flex;align-items:center;justify-content:space-between;gap:8px;padding:12px 6px 0}
.ac .tt{min-width:0}
.ac .tt b{display:block;font-size:14.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ac .tt span{display:block;font-size:12.5px;opacity:.65;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.ac a.bt{flex:none;background:var(--acc);color:#fff!important;text-decoration:none!important;font-weight:600;font-size:13px;padding:8px 14px;border-radius:999px;transition:background var(--t)}
.ac a.bt:hover{background:#2f4bc4}.ac a.bt:focus-visible{outline:2px solid var(--acc-2);outline-offset:2px}
/* Inputs */
div[data-baseweb="select"]>div,div[data-baseweb="input"],div[data-baseweb="base-input"]{border-radius:10px!important}
[data-testid="stTextInput"] input{min-height:44px}
[data-testid="stWidgetLabel"] p{font-weight:600;font-size:13.5px}
[data-testid="stFileUploaderDropzone"]{border-radius:14px;border:1.5px dashed var(--line);background:transparent;transition:border-color var(--t),background var(--t)}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--acc-2);background:var(--acc-soft)}
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--surf)}
[data-testid="stExpander"] summary{font-weight:600}
/* Segmented control */
[data-testid="stRadio"] div[role="radiogroup"]{gap:4px;background:var(--surf);padding:4px;border-radius:12px;width:fit-content;max-width:100%;flex-wrap:wrap;border:1px solid var(--line)}
[data-testid="stRadio"] label{margin:0!important;padding:8px 16px;border-radius:9px;cursor:pointer;transition:background var(--t),color var(--t)}
[data-testid="stRadio"] label>div:first-child{display:none}
[data-testid="stRadio"] label:hover{background:var(--acc-soft)}
[data-testid="stRadio"] label:has(input:checked){background:var(--acc);color:#fff}
[data-testid="stRadio"] label:has(input:checked) *{color:#fff!important}
[data-testid="stRadio"] label:has(input:focus-visible){outline:2px solid var(--acc-2);outline-offset:2px}
/* Buttons */
div.stButton>button,div.stDownloadButton>button{min-height:46px;border-radius:12px;font-weight:600;transition:background var(--t),border-color var(--t)}
div.stButton>button[kind="primary"],div.stButton>button[data-testid="stBaseButton-primary"]{background:var(--acc);border:1px solid var(--acc);color:#fff}
div.stButton>button[kind="primary"]:hover,div.stButton>button[data-testid="stBaseButton-primary"]:hover{background:#2f4bc4;border-color:#2f4bc4;color:#fff}
div.stButton>button:focus-visible,div.stDownloadButton>button:focus-visible{outline:2px solid var(--acc-2);outline-offset:2px}
div.stButton>button:disabled{opacity:.42}
[data-testid="stProgress"]>div>div>div>div{background:var(--acc)}
.bar-st{display:flex;align-items:center;gap:10px;min-height:46px;font-size:14px;flex-wrap:wrap}.bar-st .sub{opacity:.68;font-size:12.5px}
@media (prefers-reduced-motion:reduce){*{transition:none!important;animation-duration:.01ms!important}}
</style>
""", unsafe_allow_html=True)

# ---- Icon SVG (cùng bộ nét 1.8px với trang Cài đặt, không dùng emoji) ----
_ICONS = {
    "court": '<rect x="3" y="5" width="18" height="14" rx="2"/><line x1="12" y1="5" x2="12" y2="19"/><line x1="3" y1="12" x2="21" y2="12"/>',
    "users": '<circle cx="9" cy="8" r="3"/><path d="M3 20c0-3.3 2.7-6 6-6s6 2.7 6 6"/><circle cx="17" cy="9" r="2.5"/><path d="M17 14c2.5 0 4 2 4 5"/>',
    "cpu": '<rect x="6" y="6" width="12" height="12" rx="2"/><rect x="9.5" y="9.5" width="5" height="5"/><path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4"/>',
    "zap": '<path d="M13 2L4 14h7l-1 8 9-12h-7l1-8z"/>',
    "ball": '<circle cx="12" cy="12" r="9"/><path d="M5 6c4 3 4 9 0 12M19 6c-4 3-4 9 0 12"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/>',
    "tick": '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    "alert": '<path d="M12 3l10 18H2L12 3z"/><line x1="12" y1="10" x2="12" y2="15"/><line x1="12" y1="18" x2="12" y2="18.1"/>',
    "dot": '<circle cx="12" cy="12" r="4" fill="currentColor"/>',
    "upload": '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 16v3a1 1 0 0 0 1 1h14a1 1 0 0 0 1-1v-3"/>',
    "play": '<path d="M7 4l13 8-13 8V4z" fill="currentColor"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "gauge": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/>',
    "route": '<circle cx="6" cy="18" r="2"/><circle cx="18" cy="6" r="2"/><path d="M8 18h6a3 3 0 0 0 0-6h-4a3 3 0 0 1 0-6h6"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "image": '<rect x="3" y="4" width="18" height="16" rx="2"/><circle cx="9" cy="10" r="1.6"/><path d="M21 16l-5-5-8 8"/>',
    "download": '<path d="M12 4v12M7 11l5 5 5-5"/><path d="M4 20h16"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 8v4l3 2"/>',
    "list": '<line x1="8" y1="6" x2="20" y2="6"/><line x1="8" y1="12" x2="20" y2="12"/><line x1="8" y1="18" x2="20" y2="18"/><circle cx="4" cy="6" r="1" fill="currentColor"/><circle cx="4" cy="12" r="1" fill="currentColor"/><circle cx="4" cy="18" r="1" fill="currentColor"/>',
}


def svg(name, size=18):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def chip(text, kind="", icon=None):
    ic = svg(icon, 14) if icon else ""
    return f'<span class="chip {kind}">{ic}{_html.escape(str(text))}</span>'


def sec(icon, title, desc=""):
    st.markdown(
        f'<div class="sec"><div class="sec-ic">{svg(icon, 18)}</div><div>'
        f'<div class="sec-t" role="heading" aria-level="2">{title}</div>'
        f'<div class="sec-d">{desc}</div></div></div>', unsafe_allow_html=True)


def block_title(icon, text):
    st.markdown(f'<div class="blk-t" role="heading" aria-level="2">{svg(icon, 20)}{text}</div>', unsafe_allow_html=True)


def card(name):
    """Card có viền; key để CSS nhắm đúng (fallback nếu Streamlit cũ không hỗ trợ key)."""
    try:
        return st.container(border=True, key=f"card_{name}")
    except TypeError:
        return st.container(border=True)


LABELS = ["Tải file lên", "Phân tích", "Kết quả"]


def stepper_html(step):
    parts = []
    for i, label in enumerate(LABELS):
        if i < step or step == 2:
            cls, icon = "done", svg("tick", 14)
        elif i == step:
            cls, icon = "active", str(i + 1)
        else:
            cls, icon = "", str(i + 1)
        parts.append(f'<div class="stp {cls}"><i>{icon}</i>{label}</div>')
    return '<div class="stepper">' + '<div class="sbar"></div>'.join(parts) + "</div>"


def team_name(names, team, doubles):
    m = re.search(r"\d+", str(team))
    t = int(m.group(0)) if m else 1
    ids = (1, 3) if t == 1 else (2, 4)
    if not doubles:
        ids = ids[:1]
    return " / ".join(names.get(str(i), f"Player {i}") for i in ids)


def decision_html(d, names, doubles=False, rallies=None):
    if rallies and len(rallies) > 1:
        cards = []
        for r in rallies:
            out = r.get("decision") == "OUT"
            winner = _html.escape(team_name(names, r.get("point_winner") or 1, doubles))
            rid = r.get("rally_index", 1)
            trange = r.get("time_range", "")
            sf = r.get("segment_frames", [])
            f_label = f"Frame {sf[0]}–{sf[1]}" if len(sf) == 2 else ""
            title_parts = [f"Pha bóng #{rid}"]
            if trange:
                title_parts.append(f"({trange})")
            if f_label:
                title_parts.append(f"· {f_label}")
            title_str = " ".join(title_parts)

            reason = r.get("referee_reason", "")
            reason_html = f'<dt>Chi tiết</dt><dd>{_html.escape(reason)}</dd>' if reason else ""

            cards.append(
                f'<div class="dec {"out" if out else ""}" style="margin-top:10px">'
                f'<div style="flex-basis:100%;font-weight:700;font-size:14px;color:var(--txt);margin-bottom:-6px">🎾 {title_str}</div>'
                f'<div class="badge">{r.get("decision", "-")}</div>'
                f'<dl><dt>Điểm cho</dt><dd>{winner}</dd>'
                f'<dt>Vạch gần nhất</dt><dd>{_html.escape(str(r.get("nearest_line", "-")))}</dd>'
                f'<dt>Margin</dt><dd>{r.get("margin_cm", 0):+.1f} cm</dd>'
                f'<dt>Bóng chạm đất</dt><dd>Frame {r.get("landing_frame", "-")} ({r.get("landing_time_s", 0):.2f}s)</dd>'
                f'{reason_html}</dl></div>'
            )
        return (
            '<div style="margin-top:14px">'
            '<div style="font-weight:700;font-size:15px;margin-bottom:6px">⚖️ Quyết định trọng tài Hawk-Eye theo từng pha bóng:</div>'
            + "".join(cards) +
            '</div>'
        )
    if not d:
        return ""
    out = d.get("decision") == "OUT"
    winner = _html.escape(team_name(names, d.get("point_winner") or 1, doubles))
    return (f'<div class="dec {"out" if out else ""}"><div class="badge">{d.get("decision", "-")}</div>'
            f'<dl><dt>Điểm cho</dt><dd>{winner}</dd><dt>Vạch gần nhất</dt><dd>{_html.escape(str(d.get("nearest_line", "-")))}</dd>'
            f'<dt>Margin</dt><dd>{d.get("margin_cm", 0):+.1f} cm</dd>'
            f'<dt>Bóng chạm đất lần 1</dt><dd>Frame {d.get("landing_frame", "-")}</dd></dl></div>')


def kpi_html(items):
    cells = "".join(
        f'<div class="kpi"><div class="ic">{svg(ic, 20)}</div><div><div class="v">{v}<small>{u}</small></div>'
        f'<div class="l">{l}</div></div></div>' for ic, v, u, l in items)
    return f'<div class="kpis">{cells}</div>'


# =====================================================================================
# Chạy main.py (subprocess) + xử lý video
# =====================================================================================
PROGRESS_RULES = [  # (chuỗi trong log của main.py, tiến độ, nhãn)
    ("[Pipeline] Cache", 0.08, "Kiểm tra cache & mô hình"),
    ("[Court Segments]", 0.10, "Tách đoạn toàn cảnh sân"),
    ("[CourtLineDetector]", 0.45, "Dò vạch sân (CPV)"),
    ("Loaded", 0.60, "Chọn cầu thủ & chiếu mini-court"),
    ("[RefereeSystem]", 0.75, "Trọng tài Hawk-Eye"),
    ("[MatchAnalyzer] Starting", 0.80, "Phân tích sau trận"),
    ("Drawing detections", 0.88, "Vẽ & xuất video"),
    ("Successfully generated", 0.93, "Mã hoá video H.264"),
]


def deep_set(d, path, value):
    keys = path.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def make_temp_config(video_path, out_name, settings, analytics_root, with_visuals=False, match_mode="auto"):
    """Config tạm = config.yaml + config.user.yaml (trang Cài đặt), trỏ tới video đang xử lý."""
    cfg = deepcopy({k: v for k, v in settings.items() if not k.startswith("_")})  # deepcopy: không làm đổi `settings`
    use_stub = bool(cfg["tracking"].get("use_stubs", False))
    deep_set(cfg, "video.input_path", video_path.replace("\\", "/"))
    deep_set(cfg, "video.output_dir", OUT_DIR)
    deep_set(cfg, "video.output_filename", out_name)
    deep_set(cfg, "tracking.use_stubs", use_stub)
    deep_set(cfg, "tracking.force_live", not use_stub)
    deep_set(cfg, "tracking.match_mode", match_mode)
    # VIDEO SẠCH: radar / bảng thống kê / thẻ trọng tài / số frame do frontend vẽ (bật-tắt theo Cài đặt),
    # nên không vẽ cứng vào video. Người chơi / bóng / vạch sân theo đúng Cài đặt.
    cfg.setdefault("visualization", {}).update(
        {"draw_mini_court": False, "draw_stats": False, "draw_referee_decision": False,
         "draw_frame_number": False})
    # Dữ liệu cho radar/bảng thống kê lấy từ MatchAnalyzer -> bắt buộc bật analytics.
    # MatchAnalyzer ghi vào <export_dir>/<video_stem>/ ; biểu đồ PNG chỉ sinh khi người dùng tick (chậm hơn).
    cfg.setdefault("analytics", {}).update(
        {"enabled": True, "export_dir": analytics_root.replace("\\", "/"),
         "generate_visuals": bool(with_visuals)})
    path = os.path.join(TMP_DIR, f"cfg_{uuid.uuid4().hex[:8]}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def run_main(cfg_path, on_progress, log_box):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-u", "main.py", "--config", cfg_path]
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace")
    lines, best = [], 0.0
    for line in proc.stdout:
        clean_line = line.rstrip()
        print(clean_line, flush=True)  # Đẩy log trực tiếp ra terminal
        lines.append(clean_line)
        for key, frac, label in PROGRESS_RULES:
            if key in clean_line and frac > best:
                best = frac
                on_progress(frac, label)
        log_box.code("\n".join(lines[-25:]), language="text")
    proc.wait()
    return proc.returncode, "\n".join(lines)


# Đường dẫn ffmpeg.exe (ưu tiên biến môi trường FFMPEG_PATH nếu có, nếu không dùng đường dẫn mặc định dưới đây)
FFMPEG_PATH = os.environ.get(
    "FFMPEG_PATH",
    r"E:\SOFTWARE\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\ffmpeg-2026-04-09-git-d3d0b7a5ee-essentials_build\bin\ffmpeg.exe",
)


def ffmpeg_exe():
    # 1) đường dẫn chỉ định thủ công
    if FFMPEG_PATH and os.path.isfile(FFMPEG_PATH):
        return FFMPEG_PATH
    # 2) imageio-ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    # 3) ffmpeg trong PATH
    return shutil.which("ffmpeg")


def to_h264(src, dst):
    """Trình duyệt không phát được MJPG/.avi của save_video() -> chuyển sang H.264 .mp4."""
    exe = ffmpeg_exe()
    if not exe:
        return False
    r = subprocess.run([exe, "-y", "-i", src, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast",
                        "-crf", "23", "-movflags", "+faststart", dst], capture_output=True)
    return r.returncode == 0 and os.path.exists(dst)


# =====================================================================================
# Đọc kết quả MatchAnalyzer (CSV/JSON trong analytics/<stem>/) -> payload cho frontend
# -------------------------------------------------------------------------------------
# data_exporter.py quyết định tên file/cột nên ở đây dò theo từ khoá thay vì cố định tên.
# Nếu không khớp, sửa các hằng *_KEYS bên dưới (hoặc xem expander "Chi tiết đọc analytics").
# =====================================================================================
class AnalyticsError(Exception):
    pass


FRAME_KEYS = ("frame", "frame_num", "frame_idx", "frame_index", "frame_id", "shot_frame", "hit_frame", "start_frame")
ID_KEYS = ("player_id", "player", "pid", "track_id", "id")                        # file tracking dạng dài
HITTER_KEYS = ("player_id", "hitter_id", "hitter", "hit_player", "shot_player", "player")  # file shots
SPEED_KEYS = ("shot_speed_kmh", "ball_speed_kmh", "speed_kmh", "shot_speed", "ball_speed", "speed")
MAX_PLAYER_KMH = 36.0   # chặn nhiễu tracking khi tính tốc độ di chuyển


def _n(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def _col(df, keys):
    m = {}
    for c in df.columns:
        m.setdefault(_n(c), c)
    for k in keys:
        if k in m:
            return m[k]
    return None


def _axis(df, axis):
    """Tìm cột toạ độ x hoặc y; ưu tiên cột có chữ mini/court/meter."""
    pat = re.compile(rf"(^|_){axis}($|_)")
    cands = [c for c in df.columns if pat.search(_n(c))]
    if not cands:
        return None
    pref = [c for c in cands if re.search(r"mini|court|(^|_)m($|_)|meter", _n(c))]
    return (pref or cands)[0]


def _unit(name, series):
    """'m' (mét, hệ CourtGeometry) hay 'px' (pixel mini-court)."""
    n = _n(name)
    if re.search(r"(^|_)(m|meter|meters|metre|metres)(_|$)", n):
        return "m"
    if any(t in n for t in ("mini", "px", "pixel")):
        return "px"
    v = pd.to_numeric(series, errors="coerce").abs().max()
    return "m" if (pd.notna(v) and v <= 30) else "px"


def _digits(v):
    m = re.search(r"\d+", str(v))
    return int(m.group()) if m else None


def _int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        d = _digits(v)
        return d if d is not None else default


def _num(v, default=0.0):
    try:
        v = float(v)
        return v if np.isfinite(v) else default
    except (TypeError, ValueError):
        return default


def make_converter(mc):
    """(x, y, unit) -> [nx, ny] chuẩn hoá theo vùng sân của MiniCourt (0..1), hoặc None."""
    sx, sy = mc.court_start_x, mc.court_start_y
    cw, ch = float(mc.court_end_x - sx), float(mc.court_end_y - sy)
    geom = []

    def conv(x, y, unit):
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            return None
        x, y = float(x), float(y)
        if unit == "m":
            if not geom:
                try:
                    from analysis.court_geometry import CourtGeometry
                except ImportError:
                    from court_geometry import CourtGeometry
                geom.append(CourtGeometry())
            x, y = geom[0].court_meters_to_mini_px(mc, x, y)[:2]
        return [round((x - sx) / cw, 4), round((y - sy) / ch, 4)]
    return conv, cw, ch


def _frames(df):
    fcol = _col(df, FRAME_KEYS)
    s = pd.to_numeric(df[fcol], errors="coerce") if fcol else pd.Series(np.arange(len(df)), index=df.index)
    return s, fcol


def _fill(arr, n, frames, xs, ys, conv, unit):
    for f, x, y in zip(frames, xs, ys):
        if pd.notna(f) and 0 <= int(f) < n:
            arr[int(f)] = conv(x, y, unit)


def read_player_tracks(df, n, conv, dbg):
    """-> {"1": [[nx,ny]|None,...], ...}. Hỗ trợ file dạng dài (frame, player_id, x, y) và dạng rộng (p1_mini_x...)."""
    frames, fcol = _frames(df)
    dbg["player_frame_col"] = fcol or "(chỉ số dòng)"
    tracks = {}
    icol, xcol, ycol = _col(df, ID_KEYS), _axis(df, "x"), _axis(df, "y")
    if icol and xcol and ycol:
        unit = _unit(xcol, df[xcol])
        dbg["player_format"] = f"dài: id={icol}, x={xcol}, y={ycol}, đơn vị={unit}"
        for pid, g in df.groupby(icol):
            k = _digits(pid)
            if k is None:
                continue
            arr = [None] * n
            _fill(arr, n, frames[g.index], g[xcol], g[ycol], conv, unit)
            tracks[str(k)] = arr
    else:
        cols = {}
        for c in df.columns:
            m = re.match(r"^(?:p|player_?)(\d+)_(.+)$", _n(c))
            if not m:
                continue
            for ax in ("x", "y"):
                if re.search(rf"(^|_){ax}($|_)", m.group(2)):
                    cols.setdefault(m.group(1), {}).setdefault(ax, c)
        used = []
        for pid, d in cols.items():
            if "x" in d and "y" in d:
                unit = _unit(d["x"], df[d["x"]])
                arr = [None] * n
                _fill(arr, n, frames, df[d["x"]], df[d["y"]], conv, unit)
                tracks[pid] = arr
                used.append(f"p{pid}:{d['x']}/{d['y']}({unit})")
        dbg["player_format"] = "rộng: " + ", ".join(used)
    if not tracks:
        raise AnalyticsError("Không nhận ra cột toạ độ cầu thủ trong file player tracking.")
    return tracks


def read_ball(df, n, conv, dbg):
    frames, fcol = _frames(df)
    xcol, ycol = _axis(df, "x"), _axis(df, "y")
    if not (xcol and ycol):
        raise AnalyticsError("File ball tracking không có cột toạ độ x/y.")
    unit = _unit(xcol, df[xcol])
    dbg["ball_format"] = f"frame={fcol or '(chỉ số dòng)'}, x={xcol}, y={ycol}, đơn vị={unit}"
    arr = [None] * n
    _fill(arr, n, frames, df[xcol], df[ycol], conv, unit)
    return arr


def read_shots(df, dbg, warns):
    """-> list (frame, player_id|None, speed_kmh)."""
    df = df.reset_index(drop=True)
    frames, fcol = _frames(df)
    if not fcol:
        warns.append("File shots: không tìm thấy cột frame -> không có marker cú đánh / tốc độ cú đánh.")
        return []
    hcol = _col(df, HITTER_KEYS)
    scol = _col(df, SPEED_KEYS) or next(
        (c for c in df.columns if "speed" in _n(c) and "km" in _n(c)), None)
    dbg["shots_cols"] = f"frame={fcol}, hitter={hcol}, speed={scol}"
    if not scol:
        warns.append("File shots: không tìm thấy cột tốc độ (km/h) -> tốc độ cú đánh = 0.")
    if not hcol:
        warns.append("File shots: không tìm thấy cột người đánh -> giả định hai bên đánh luân phiên.")
    events = []
    for i, f in frames.items():
        if pd.isna(f):
            continue
        pid = _digits(df.at[i, hcol]) if hcol else None
        events.append((int(f), pid, _num(df.at[i, scol]) if scol else 0.0))
    events.sort(key=lambda e: e[0])
    return events


def _team_of(pid, k):
    if pid is None:                       # không biết người đánh -> luân phiên
        return 1 if k % 2 == 0 else 2
    return 1 if pid % 2 == 1 else 2       # P1,P3 = đội 1 ; P2,P4 = đội 2


def compute_stats(n, fps, events, tracks, ratio):
    """Bảng thống kê theo frame: [last_shot1, last_shot2, player_spd1, player_spd2,
    avg_shot1, avg_shot2, avg_player_spd1, avg_player_spd2] (đúng thứ tự s0..s7 của frontend)."""
    import constants
    dlw = constants.DOUBLE_LINE_WIDTH
    # --- tốc độ cú đánh
    last, tot, cnt, k = [0.0, 0.0], [0.0, 0.0], [0, 0], 0
    shot = np.zeros((n, 4))
    for f in range(n):
        while k < len(events) and events[k][0] <= f:
            t = _team_of(events[k][1], k) - 1
            last[t] = events[k][2]
            tot[t] += events[k][2]
            cnt[t] += 1
            k += 1
        shot[f] = [last[0], last[1], tot[0] / cnt[0] if cnt[0] else 0.0, tot[1] / cnt[1] if cnt[1] else 0.0]
    # --- tốc độ di chuyển (cửa sổ ~0.5s) + tổng quãng đường
    w = max(1, int(round(fps / 2)))
    spd = {1: np.zeros(n), 2: np.zeros(n)}
    dist_m = 0.0
    for pid, pts in tracks.items():
        a = pd.DataFrame([[np.nan, np.nan] if p is None else p for p in pts]).ffill().bfill().to_numpy(dtype=float)
        if a.size == 0 or np.isnan(a).any():
            continue
        dist_m += float(np.hypot(np.diff(a[:, 0]) * dlw, np.diff(a[:, 1]) * ratio * dlw).sum())
        d = np.zeros(n)
        if n > w:
            dx, dy = (a[w:, 0] - a[:-w, 0]) * dlw, (a[w:, 1] - a[:-w, 1]) * ratio * dlw
            d[w:] = np.hypot(dx, dy) / (w / fps) * 3.6
        t = 1 if int(pid) % 2 == 1 else 2
        spd[t] = np.maximum(spd[t], np.clip(d, 0, MAX_PLAYER_KMH))
    idx = np.arange(1, n + 1)
    avg = {t: np.cumsum(spd[t]) / idx for t in (1, 2)}
    stats = np.column_stack([shot[:, 0], shot[:, 1], spd[1], spd[2],
                             shot[:, 2], shot[:, 3], avg[1], avg[2]])
    return np.round(stats, 1).tolist(), dist_m


def _find_decision(o):
    if isinstance(o, dict):
        d = o.get("decision")
        if isinstance(d, str) and d.strip().upper() in ("IN", "OUT", "NET", "WINNER"):
            return o
        o = list(o.values())
    if isinstance(o, list):
        for v in o:
            r = _find_decision(v)
            if r:
                return r
    return None


def _pick(d, *keys, default=None):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def _find_file(paths, need, any_of=(), root_dir=None):
    def sort_key(p):
        is_root = 0 if root_dir and os.path.dirname(os.path.abspath(p)) == os.path.abspath(root_dir) else 1
        return (is_root, len(p), p)
    for p in sorted(paths, key=sort_key):
        b = _n(os.path.basename(p))
        if all(k in b for k in need) and (not any_of or any(k in b for k in any_of)):
            return p
    return None


def describe_dir(adir):
    """Liệt kê file + cột/khoá trong thư mục analytics (hiện khi không nhận ra cấu trúc)."""
    lines = []
    for p in sorted(glob.glob(os.path.join(adir, "**", "*"), recursive=True)):
        if os.path.isdir(p):
            continue
        rel = os.path.relpath(p, adir)
        try:
            if p.lower().endswith(".csv"):
                lines.append(f"{rel}: cột = {list(pd.read_csv(p, nrows=2).columns)}")
            elif p.lower().endswith(".json"):
                with open(p, "r", encoding="utf-8") as f:
                    o = json.load(f)
                lines.append(f"{rel}: khoá = {list(o) if isinstance(o, dict) else type(o).__name__}")
            else:
                lines.append(rel)
        except Exception as e:  # noqa: BLE001
            lines.append(f"{rel}: (không đọc được: {e})")
    return "\n".join(lines) or "(thư mục trống)"


def load_analytics_payload(input_video, out_video_for_fps, adir, names):
    """Đọc analytics/<stem>/ -> (payload, warns, found_files, debug_info)."""
    from mini_court import MiniCourt

    warns, dbg = [], {}
    csvs = glob.glob(os.path.join(adir, "**", "*.csv"), recursive=True)
    jsons = glob.glob(os.path.join(adir, "**", "*.json"), recursive=True)
    found = {
        "player": _find_file(csvs, ("player",), ("track", "pos", "traj"), root_dir=adir),
        "ball": _find_file(csvs, ("ball",), ("track", "pos", "traj"), root_dir=adir),
        "shots": _find_file(csvs, ("shot",), root_dir=adir),
        "summary": (_find_file(jsons, ("summary",), root_dir=adir) or _find_file(jsons, ("kpi",), root_dir=adir)
                    or (sorted(jsons, key=lambda p: (0 if os.path.dirname(os.path.abspath(p)) == os.path.abspath(adir) else 1, p))[0] if jsons else None)),
    }
    dbg["files"] = {k: (os.path.relpath(v, adir) if v else None) for k, v in found.items()}
    missing = [k for k in ("player", "ball") if not found[k]]
    if missing:
        raise AnalyticsError(f"Không tìm thấy file tracking: {', '.join(missing)} "
                             f"(tìm theo tên chứa 'player'/'ball' + 'track'/'pos').")

    cap = cv2.VideoCapture(input_video)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    cap = cv2.VideoCapture(out_video_for_fps)
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0        # fps của VIDEO XUẤT -> khớp radar
    cap.release()

    mc = MiniCourt(np.zeros((h, w, 3), dtype=np.uint8))
    conv, cw, ch = make_converter(mc)
    nx = lambda x: round((x - mc.court_start_x) / cw, 4)   # noqa: E731
    ny = lambda y: round((y - mc.court_start_y) / ch, 4)   # noqa: E731

    p_df, b_df = pd.read_csv(found["player"]), pd.read_csv(found["ball"])
    if n <= 0:
        n = int(max(len(p_df), len(b_df)))
    tracks = read_player_tracks(p_df, n, conv, dbg)
    ball = read_ball(b_df, n, conv, dbg)

    events = []
    if found["shots"]:
        events = read_shots(pd.read_csv(found["shots"]), dbg, warns)
    else:
        warns.append("Không thấy file shots (tên chứa 'shot') -> không có marker cú đánh / tốc độ cú đánh.")

    stats, dist_m = compute_stats(n, fps, events, tracks, ch / cw)

    decision = None
    rallies = []
    segment_count = 1
    if found["summary"]:
        try:
            with open(found["summary"], "r", encoding="utf-8") as f:
                s_data = json.load(f)
                rd = _find_decision(s_data)
                segment_count = s_data.get("segment_count", 1)
                if "rallies" in s_data and isinstance(s_data["rallies"], list):
                    for r in s_data["rallies"]:
                        lp = _pick(r, "landing_pos_mini", "first_bounce_pos", "landing_pos")
                        rallies.append({
                            "rally_index": r.get("rally_index", 1),
                            "segment_frames": r.get("segment_frames", []),
                            "time_range": r.get("time_range", ""),
                            "decision": str(r.get("decision", "IN")).strip().upper(),
                            "point_winner": _int(_pick(r, "point_winner", "winner", default=1)),
                            "margin_cm": _num(_pick(r, "margin_cm", "margin", default=0)),
                            "nearest_line": str(_pick(r, "nearest_line", "line", default="")),
                            "landing_frame": _int(_pick(r, "landing_frame", "bounce_frame", "frame", default=0)),
                            "landing_time_s": _num(r.get("landing_time_s", 0.0)),
                            "landing_pos": (conv(lp[0], lp[1], "px") if isinstance(lp, (list, tuple)) and len(lp) >= 2 else None),
                            "referee_reason": str(r.get("referee_reason", "")),
                            "shots_count": int(r.get("shots_count", 0)),
                        })
        except Exception as e:  # noqa: BLE001
            rd = None
            warns.append(f"Không đọc được summary JSON: {e}")
        if rd:
            lp = _pick(rd, "landing_pos_mini", "first_bounce_pos", "landing_pos")
            decision = {
                "decision": str(rd["decision"]).strip().upper(),
                "margin_cm": _num(_pick(rd, "margin_cm", "margin", default=0)),
                "nearest_line": _pick(rd, "nearest_line", "line"),
                "point_winner": _int(_pick(rd, "point_winner", "winner")),
                "hitter_id": _int(_pick(rd, "hitter_id", "hitter")),
                "landing_frame": _int(_pick(rd, "landing_frame", "bounce_frame", "frame")),
                "landing_pos": (conv(lp[0], lp[1], "px") if isinstance(lp, (list, tuple)) and len(lp) >= 2 else None),
            }
        else:
            warns.append("Summary JSON không có quyết định trọng tài (decision = IN/OUT) -> bỏ qua thẻ IN/OUT.")
    else:
        warns.append("Không thấy summary JSON -> bỏ qua thẻ IN/OUT của trọng tài.")
    dbg["decision_found"] = decision is not None
    dbg["rallies_count"] = len(rallies)

    kc = mc.drawing_key_points
    court = [[nx(kc[2 * i]), ny(kc[2 * i + 1])] for i in range(14)]
    shots = sorted({e[0] for e in events})
    speeds = [e[2] for e in events]
    metrics = {"max_shot_kmh": round(max(speeds), 1) if speeds else 0.0,
               "distance_m": round(dist_m, 1), "hits": len(shots),
               "ball_detect_rate": sum(b is not None for b in ball) / max(1, n)}
    payload = {"fps": float(fps), "n": n, "w": w, "h": h, "court": court, "ball": ball,
               "players": tracks, "stats": stats, "shots": shots, "decision": decision,
               "rallies": rallies, "segment_count": segment_count,
               "names": names, "metrics": metrics}
    return payload, warns, found, dbg


# =====================================================================================
# Component HTML: video sạch + radar Hawk-Eye + bảng thống kê (đồng bộ theo currentTime)
# =====================================================================================
PLAYER_HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8"><style>
*{box-sizing:border-box}body{margin:0;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#e8ecf8}
.g{display:grid;grid-template-columns:minmax(0,1fr) 290px;gap:16px}
@media (max-width:760px){.g{grid-template-columns:minmax(0,1fr)}}
.card,.radar{background:#121829;border:1px solid rgba(128,140,180,.28);border-radius:18px;padding:12px}
.vw{position:relative;background:#000;border-radius:12px;overflow:hidden;width:100%;max-width:1200px;margin:0 auto}
.vw video{position:absolute;inset:0;width:100%;height:100%}
.ctrl{display:flex;align-items:center;gap:12px;margin-top:12px;font-size:13px;flex-wrap:wrap;font-variant-numeric:tabular-nums}
.ctrl button{width:40px;height:40px;display:grid;place-items:center;background:#3b5bdb;color:#fff;border:0;border-radius:10px;cursor:pointer;transition:background .16s ease}
.ctrl button:hover{background:#2f4bc4}
.ctrl select{height:40px;background:#1c2540;color:#e8ecf8;border:1px solid rgba(128,140,180,.35);border-radius:10px;padding:0 10px;cursor:pointer}
.ctrl button:focus-visible,.ctrl select:focus-visible,.tl input:focus-visible{outline:2px solid #5c7cfa;outline-offset:2px}
.tl{position:relative;flex:1;min-width:140px;height:24px}
.tl input{position:absolute;inset:0 0 auto 0;width:100%;margin:0;height:22px;z-index:2;background:transparent;accent-color:#5c7cfa;cursor:pointer}
.mk{position:absolute;top:17px;width:6px;height:6px;border-radius:50%;background:#f59f00;transform:translateX(-3px)}
.radar{margin-bottom:14px}
.radar h3{margin:0 0 8px;font-size:12px;letter-spacing:.1em;display:flex;justify-content:space-between;font-weight:700}
.radar h3 em{color:#4ade80;font-style:normal}.radar canvas{width:100%;display:block;border-radius:10px}
.lg{display:flex;flex-wrap:wrap;gap:6px 14px;font-size:12px;margin-top:10px;color:#aab3cf}
.lg span:before{content:"";display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px;background:var(--c)}
table{width:100%;border-collapse:separate;border-spacing:0;font-size:14px;background:#121829;color:#e8ecf8;border:1px solid rgba(128,140,180,.28);border-radius:18px;overflow:hidden;font-variant-numeric:tabular-nums}
th,td{padding:11px 12px;text-align:right;border-top:1px solid rgba(128,140,180,.16)}
th{border-top:0;background:#1c2540;font-size:13px}
th:first-child,td:first-child{text-align:left;color:#aab3cf;font-size:13px}
</style></head><body><div class="g">
<div class="card"><div class="vw" id="vw"><video id="v" playsinline muted preload="auto"></video></div>
<div class="ctrl"><button id="pl" aria-label="Phát / tạm dừng"><svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4l13 8-13 8V4z"/></svg></button><span id="tm">0:00</span>
<div class="tl"><input type="range" id="sk" min="0" max="1000" value="0" aria-label="Thanh thời gian video"><div id="mks"></div></div>
<select id="sp" aria-label="Tốc độ phát"><option>0.5</option><option selected>1</option><option>2</option></select></div></div>
<div><div class="radar"><h3><span>HAWK-EYE 2D RADAR</span><em>● LIVE</em></h3>
<canvas id="rd" width="260" height="440"></canvas>
<div class="lg" id="lg"></div></div>
<table><tr><th></th><th id="n1">Player 1</th><th id="n2">Player 2</th></tr>
<tr><td>Shot speed</td><td id="s0">0.0</td><td id="s1">0.0</td></tr>
<tr><td>Player speed</td><td id="s2">0.0</td><td id="s3">0.0</td></tr>
<tr><td>avg. S. speed</td><td id="s4">0.0</td><td id="s5">0.0</td></tr>
<tr><td>avg. P. speed</td><td id="s6">0.0</td><td id="s7">0.0</td></tr></table></div></div>
<script>
const P=__PAYLOAD__,$=id=>document.getElementById(id),v=$('v'),rd=$('rd'),rx=rd.getContext('2d');
v.src="__VIDEO_URL__";$('vw').style.aspectRatio=P.w+'/'+P.h;
const SH=P.show||{radar:true,stats:true};
if(!SH.radar)document.querySelector('.radar').style.display='none';
if(!SH.stats)document.querySelector('table').style.display='none';
if(!SH.radar&&!SH.stats)document.querySelector('.g').style.gridTemplateColumns='minmax(0,1fr)';

const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const NM=P.names||{},has=id=>!!(P.players[id]&&P.players[id].some(Boolean)),nm=id=>NM[id]||("Player "+id),
T1=[nm("1")].concat(has("3")?[nm("3")]:[]).join(" / "),
T2=[nm("2")].concat(has("4")?[nm("4")]:[]).join(" / ");
$('n1').textContent=T1;$('n2').textContent=T2;
$('lg').innerHTML=[["1","#4aa8ff"],["3","#8fd0ff"],["2","#ff6b4a"],["4","#ffa58a"]]
 .filter(([id])=>id<3||has(id)).map(([id,c])=>`<span style="--c:${c}">${esc(nm(id))}</span>`).join("")
 +'<span style="--c:#d6ff3a">Bóng</span>';

const dur=()=>v.duration||P.n/P.fps,fmt=s=>Math.floor(s/60)+':'+String(Math.floor(s%60)).padStart(2,'0');
P.shots.forEach(f=>{const m=document.createElement('div');m.className='mk';m.style.left=(f/P.n*100)+'%';$('mks').appendChild(m);});
const IP='<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M7 4l13 8-13 8V4z"/></svg>',IU='<svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M6 4h4v16H6zM14 4h4v16h-4z"/></svg>';$('pl').onclick=()=>{v.paused?v.play():v.pause()};v.onplay=()=>$('pl').innerHTML=IU;v.onpause=()=>$('pl').innerHTML=IP;$('sp').onchange=()=>v.playbackRate=parseFloat($('sp').value);$('sk').oninput=e=>{v.currentTime=e.target.value/1000*dur()};
const pad=22,W=rd.width,H=rd.height,cw=W-2*pad,ch=H-2*pad,X=x=>pad+x*cw,Y=y=>pad+y*ch;
const LINES=[[0,1],[2,3],[0,2],[1,3],[4,5],[6,7],[8,9],[10,11],[12,13]];
function radar(f){
rx.fillStyle='#1b3f78';rx.fillRect(0,0,W,H);const c=P.court;
rx.fillStyle='#2457a3';rx.fillRect(X(c[4][0]),Y(0),(c[6][0]-c[4][0])*cw,ch);
rx.strokeStyle='#fff';rx.lineWidth=2;
LINES.forEach(([a,b])=>{rx.beginPath();rx.moveTo(X(c[a][0]),Y(c[a][1]));rx.lineTo(X(c[b][0]),Y(c[b][1]));rx.stroke();});
const ny=(c[8][1]+c[10][1])/2;rx.strokeStyle='#ffb020';rx.lineWidth=3;rx.beginPath();rx.moveTo(X(0),Y(ny));rx.lineTo(X(1),Y(ny));rx.stroke();
rx.fillStyle='#fff';rx.font='10px sans-serif';rx.fillText('NET',W/2-9,Y(ny)-6);
rx.strokeStyle='rgba(214,255,58,.6)';rx.lineWidth=2;rx.beginPath();let on=false;
for(let k=10;k>=0;k--){const b=P.ball[Math.max(0,f-k)];if(!b)continue;on?rx.lineTo(X(b[0]),Y(b[1])):rx.moveTo(X(b[0]),Y(b[1]));on=true;}rx.stroke();
const rlist=(P.rallies&&P.rallies.length>0)?P.rallies:(P.decision?[P.decision]:[]);
rlist.forEach(r=>{
if(r&&r.landing_pos&&f>=r.landing_frame){
const isActive=!r.segment_frames||(f>=r.segment_frames[0]&&f<=r.segment_frames[1]+50);
rx.save();if(!isActive&&rlist.length>1)rx.globalAlpha=0.5;
const p=r.landing_pos,ok=r.decision!=='OUT';
rx.fillStyle=ok?'#22b45a':'#e5484d';rx.beginPath();rx.ellipse(X(p[0]),Y(p[1]),8,5,0,0,7);rx.fill();
rx.strokeStyle='#fff';rx.lineWidth=1.5;rx.stroke();rx.fillStyle='#fff';
const tag=(rlist.length>1?'R'+(r.rally_index||'')+' ':'')+(ok?'IN ':'OUT ')+Math.abs(r.margin_cm||0).toFixed(0)+'cm';
rx.fillText(tag,Math.min(X(p[0])+12,W-75),Y(p[1])-6);rx.restore();}
});
[['1','#4aa8ff'],['3','#8fd0ff'],['2','#ff6b4a'],['4','#ffa58a']].forEach(([id,col])=>{
const a=P.players[id],p=a&&a[f];if(!p)return;rx.fillStyle=col;rx.beginPath();rx.arc(X(p[0]),Y(p[1]),8,0,7);rx.fill();
rx.strokeStyle='#fff';rx.lineWidth=1;rx.stroke();rx.fillStyle='#fff';rx.font='bold 9px sans-serif';rx.fillText('P'+id,X(p[0])-6,Y(p[1])+3);});
const b=P.ball[f];if(b){rx.fillStyle='#d6ff3a';rx.beginPath();rx.arc(X(b[0]),Y(b[1]),5,0,7);rx.fill();}}
function fit(){const g=document.querySelector('.g');if(!g)return;const h=Math.ceil(g.getBoundingClientRect().height)+6;
try{const fe=window.frameElement;if(fe&&Math.abs((parseInt(fe.style.height)||0)-h)>2){fe.style.height=h+'px';fe.height=h;}}catch(e){}}
v.addEventListener('loadedmetadata',fit);window.addEventListener('resize',fit);setInterval(fit,400);fit();
function loop(){const f=Math.max(0,Math.min(P.n-1,Math.floor(v.currentTime*P.fps)));radar(f);
const s=P.stats[f]||[];for(let i=0;i<8;i++)$('s'+i).textContent=(s[i]||0).toFixed(1);
$('tm').textContent=fmt(v.currentTime)+' / '+fmt(dur());$('sk').value=Math.round(v.currentTime/dur()*1000);
requestAnimationFrame(loop);}loop();
</script></body></html>"""


def render_player(payload, video_url, height=800):
    html = PLAYER_HTML.replace("__PAYLOAD__", json.dumps(payload).replace("</", "<\\/")).replace("__VIDEO_URL__", video_url)
    components.html(html, height=height, scrolling=False)


# =====================================================================================
# Giao diện
# =====================================================================================
# =====================================================================================
# Lịch sử phiên (history/sessions.csv) -> dữ liệu cho trang Dashboard
# =====================================================================================
def build_history_record(res):
    """Gom các trường cần lưu cho 1 phiên từ kết quả phân tích."""
    p = res["payload"]
    mt = p["metrics"]
    d = p.get("decision") or {}
    winner = ""
    if d:
        winner = p["names"].get(str(d.get("point_winner")), f"Player {d.get('point_winner')}")
    return {
        "session_id": res["session_id"],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "video_name": res["video_name"],
        "duration_sec": round(p["n"] / max(p["fps"], 1e-6), 2),
        "max_shot_kmh": mt["max_shot_kmh"],
        "distance_m": mt["distance_m"],
        "hits": mt["hits"],
        "decision": d.get("decision", ""),
        "point_winner": winner,
        "nearest_line": d.get("nearest_line") or "",
        "margin_cm": round(_num(d.get("margin_cm")), 1) if d else "",
        "landing_frame": d.get("landing_frame", "") if d else "",
    }


def save_history(record):
    """Ghi thêm 1 dòng vào sessions.csv (tạo file + header nếu chưa có). Trả về False nếu session đã tồn tại."""
    os.makedirs(HISTORY_DIR, exist_ok=True)
    exists = os.path.exists(HISTORY_PATH)
    if exists:
        old = pd.read_csv(HISTORY_PATH, usecols=["session_id"])
        if record["session_id"] in set(old["session_id"].astype(str)):
            return False
    pd.DataFrame([record], columns=HISTORY_COLS).to_csv(
        HISTORY_PATH, mode="a", header=not exists, index=False,
        encoding="utf-8" if exists else "utf-8-sig")   # BOM chỉ ghi 1 lần ở đầu file (Excel đọc được tiếng Việt)
    return True


ss = st.session_state
ss.setdefault("result", None)

settings = cfgu.load_config(CONFIG_PATH)   # config.yaml + config.user.yaml (trang Cài đặt)
_trk, _vis, _ply = settings["tracking"], settings["visualization"], settings["players"]
match_mode = _trk["match_mode"]
use_stub = bool(_trk["use_stubs"])
n1, n2 = _ply["player1_name"], _ply["player2_name"]

# ---------------------------------------------------------------- Header
h_l, h_r = st.columns([4, 1.3], vertical_alignment="center")
h_l.markdown(
    f'<div class="pg-hd"><div class="pg-ic">{svg("ball", 26)}</div><div>'
    '<h1 class="pg-t">Phân tích video</h1>'
    '<p class="pg-s">Tải video trận đấu để theo dõi cầu thủ, bóng, vạch sân và nhận phán quyết Hawk-Eye.</p>'
    '</div></div>', unsafe_allow_html=True)
h_r.page_link("pages/3_setting.py", label="Cài đặt", width='stretch')
if not st.get_option("server.enableStaticServing"):
    st.warning("Chưa bật static serving: tạo `.streamlit/config.toml` với\n\n"
               "`[server]\\nenableStaticServing = true` rồi chạy lại app.")
st.markdown(
    '<div class="meta">' +
    chip(f"Cache: {'bật' if use_stub else 'tắt'}", "acc", "zap") +
    chip(f"Vạch sân: {'bật' if _vis['draw_court_keypoints'] else 'tắt'}", "acc", "court") +
    chip(f"Bóng: {_trk['ball_detector']}", "acc", "ball") +
    chip(f"Thiết bị: {str(_trk['device']).upper()}", "acc", "cpu") + '</div>', unsafe_allow_html=True)
step_ph = st.empty()
step_ph.markdown(stepper_html(2 if ss.result else 0), unsafe_allow_html=True)

# ---------------------------------------------------------------- Video đầu vào
video_path = None
with card("source"):
    sec("film", "Video đầu vào", "Tải lên video trận đấu hoặc chọn clip có sẵn trong input_videos/.")
    src_mode = st.radio("Nguồn video", ["Tải lên", "Clip có sẵn"], horizontal=True, key="src_mode")
    if src_mode == "Tải lên":
        up = st.file_uploader("Tải video trận đấu", type=["mp4", "mov", "avi", "mkv"], label_visibility="collapsed")
        if up is not None:
            video_path = os.path.join(UPLOAD_DIR, os.path.basename(up.name))
            with open(video_path, "wb") as f:
                f.write(up.getbuffer())
    else:
        clips = sorted(glob.glob(CLIPS_GLOB))
        if clips:
            video_path = st.selectbox("Chọn clip", clips, label_visibility="collapsed")
        else:
            st.info(f"Không thấy clip nào ở {CLIPS_GLOB}")
    if video_path:
        st.markdown(chip(f"Đã chọn: {os.path.basename(video_path)}", "ok", "check"), unsafe_allow_html=True)
    else:
        st.markdown(chip("Chưa chọn video", "warn", "alert"), unsafe_allow_html=True)

MODE_LABELS = {"auto": "Tự nhận diện (đơn/đôi)", "singles": "Đánh đơn (2 người)", "doubles": "Đánh đôi (4 người)"}
_modes = list(MODE_LABELS)
_def_mode = _trk.get("match_mode", "auto")
DEFAULT_NAMES = {i: _ply.get(f"player{i}_name", f"Player {i}") for i in (1, 2, 3, 4)}

res = ss.result
detected_doubles = bool(res) and any(k in res["payload"]["players"] for k in ("3", "4"))

# ---------------------------------------------------------------- Thiết lập trận đấu
with card("match"):
    sec("users", "Thiết lập trận đấu", "Chế độ trận và tên cầu thủ hiển thị trên radar, bảng thống kê.")
    match_mode = st.radio("Chế độ trận", _modes, index=_modes.index(_def_mode) if _def_mode in _modes else 0,
                          format_func=MODE_LABELS.get, horizontal=True, key="match_mode")
    show_pair = match_mode == "doubles" or detected_doubles
    if detected_doubles:
        st.info("Đã nhận diện trận ĐÔI — nhập tên Player 3, 4 bên dưới, kết quả sẽ cập nhật ngay, không cần xử lý lại.")

    ca, cb = st.columns(2)
    nm_in = {}
    nm_in[1] = ca.text_input("Player 1 (đội gần camera)" if show_pair else "Player 1 (gần camera)",
                             value=DEFAULT_NAMES[1], key="pname_1")
    nm_in[2] = cb.text_input("Player 2 (đội xa camera)" if show_pair else "Player 2 (xa camera)",
                             value=DEFAULT_NAMES[2], key="pname_2")
    if show_pair:
        nm_in[3] = ca.text_input("Player 3 (đội gần camera)", value=DEFAULT_NAMES[3], key="pname_3")
        nm_in[4] = cb.text_input("Player 4 (đội xa camera)", value=DEFAULT_NAMES[4], key="pname_4")
    elif match_mode == "auto":
        with st.expander("Nếu là trận đôi: nhập thêm tên Player 3 & 4"):
            e1, e2 = st.columns(2)
            nm_in[3] = e1.text_input("Player 3 (đội gần camera)", value=DEFAULT_NAMES[3], key="pname_3")
            nm_in[4] = e2.text_input("Player 4 (đội xa camera)", value=DEFAULT_NAMES[4], key="pname_4")
names = {str(i): (v.strip() or DEFAULT_NAMES[i]) for i, v in nm_in.items()}

# ---------------------------------------------------------------- Thanh hành động (1 nút chính)
with card("actions"):
    a_l, a_r = st.columns([4, 1.6], vertical_alignment="center")
    with a_l:
        want_vis = st.toggle("Xuất thêm biểu đồ phân tích (PNG)", value=False, key="want_vis")
        st.markdown('<div class="hint" style="margin:-6px 0 0">Tạo heatmap, quỹ đạo bóng, điểm nảy... Quá trình xử lý sẽ chậm hơn.</div>',
                    unsafe_allow_html=True)
    go = a_r.button("Bắt đầu xử lý", type="primary", disabled=video_path is None, width='stretch')

if go and video_path:
    step_ph.markdown(stepper_html(1), unsafe_allow_html=True)
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_name = f"{stem}_analysis.avi"
    adir = os.path.join(ANALYTICS_ROOT, stem)            # MatchAnalyzer ghi vào <export_dir>/<video_stem>/
    shutil.rmtree(adir, ignore_errors=True)               # tránh đọc nhầm kết quả của lần chạy trước
    cfg_path = make_temp_config(video_path, out_name, settings, ANALYTICS_ROOT, want_vis, match_mode)

    bar, txt = st.progress(0.0), st.empty()
    with st.expander("Log pipeline", expanded=False):
        log_box = st.empty()

    def on_progress(frac, label):
        bar.progress(frac)
        txt.caption(f"{label}… {int(frac * 100)}%")

    code, log = run_main(cfg_path, on_progress, log_box)
    
    log_file_path = os.path.join(TMP_DIR, f"{stem}_pipeline.log")
    with open(log_file_path, "w", encoding="utf-8") as lf:
        lf.write(log)
    ss.result = None
    if code != 0:
        st.error("main.py báo lỗi — xem phần log bên dưới.")
        st.code("\n".join(log.splitlines()[-40:]), language="text")
    else:
        saved = re.search(r"analysis video at (.+?) ===", log)
        avi = (saved.group(1).strip() if saved else os.path.join(OUT_DIR, out_name))
        mp4 = os.path.join(OUT_DIR, os.path.splitext(os.path.basename(avi))[0] + ".mp4")
        if not os.path.exists(avi):
            st.error(f"Không thấy video đầu ra: {avi}")
            st.code("\n".join(log.splitlines()[-20:]), language="text")
        elif not (os.path.isdir(adir) and os.listdir(adir)):
            st.error(f"main.py không xuất analytics vào `{adir}` "
                     "(kiểm tra `analytics.enabled` trong config.yaml).")
        else:
            on_progress(0.96, "Mã hoá H.264")
            if to_h264(avi, mp4):
                try:
                    os.remove(avi)
                except OSError:
                    pass
                try:
                    payload, warns, found, dbg = load_analytics_payload(video_path, mp4, adir, names)
                    payload["show"] = {"radar": bool(_vis["draw_mini_court"]), "stats": bool(_vis["draw_stats"]),
                                       "referee": bool(_vis["draw_referee_decision"])}
                    zip_path = shutil.make_archive(os.path.join(TMP_DIR, f"{stem}_analytics"), "zip", adir)
                    ss.result = {"mp4": mp4, "adir": adir, "zip": zip_path, "found": found, "dbg": dbg,
                                 "warns": warns, "payload": payload,
                                 "log_path": log_file_path,
                                 # video nhiều góc quay: main.py phân tích từng đoạn, thư mục analytics chỉ giữ đoạn cuối
                                 "segmented": "[Court Segments] Analyzing frames" in log,
                                 "session_id": f"{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}",
                                 "video_name": os.path.basename(video_path),
                                 "saved": False}
                except AnalyticsError as e:
                    st.error(f"Không đọc được kết quả analytics: {e}")
                    st.code(describe_dir(adir), language="text")
                except Exception as e:
                    st.exception(e)
            else:
                st.error("Không mã hoá được H.264. Cài `pip install imageio-ffmpeg` (hoặc cài ffmpeg vào PATH).")
    bar.empty(); txt.empty()
    try:
        os.remove(cfg_path)
    except OSError:
        pass
    step_ph.markdown(stepper_html(2 if ss.result else 0), unsafe_allow_html=True)
    if ss.result:
        st.rerun()

# Tên hiển thị + mô tả ngắn cho từng biểu đồ analytics (khoá = tên file PNG do MatchVisualizer xuất ra)
ANALYSIS_META = {
    "heatmap_player_1.png": ("Heatmap Player 1", "Vùng phủ sân · sân gần"),
    "heatmap_player_2.png": ("Heatmap Player 2", "Vùng phủ sân · sân xa"),
    "heatmap_combined.png": ("Heatmap tổng hợp", "Player 1 vs Player 2"),
    "court_zones_distribution.png": ("Phân bổ vùng sân", "Baseline / Giữa sân / Lưới"),
    "ball_bounce_dispersion.png": ("Điểm bóng nảy", "Hawk-Eye · IN / OUT"),
    "ball_trajectories_2d.png": ("Quỹ đạo bóng 2D", "Vector từng cú đánh"),
    "speed_and_distance_timeline.png": ("Tốc độ theo thời gian", "Bóng & người chơi"),
    "match_summary_dashboard.png": ("Tổng quan pha bóng", "Thẻ tóm tắt"),
}


def analysis_cards_html(pngs, session_id):
    """Lưới thẻ ảnh thu nhỏ (thumbnail cố định 4:3). Bấm ảnh / nút «Xem ảnh» để mở bản đầy đủ ở tab mới.
    PNG được copy sang OUT_DIR (cùng thư mục static với video) để trình duyệt tải trực tiếp, không nhúng base64."""
    import html as _html
    cards = []
    for png in pngs:
        base = os.path.basename(png)
        dst_name = f"{session_id}_{base}"
        dst = os.path.join(OUT_DIR, dst_name)
        try:
            if not os.path.exists(dst) or os.path.getmtime(dst) < os.path.getmtime(png):
                shutil.copy2(png, dst)
        except OSError:
            continue
        title, sub = ANALYSIS_META.get(base, (os.path.splitext(base)[0].replace("_", " ").title(), "Biểu đồ phân tích"))
        url = f"/app/static/outputs/{dst_name}"
        cards.append(
            '<div class="ac">'
            f'<a class="th" href="{url}" target="_blank" rel="noopener"><img src="{url}" loading="lazy" alt="{_html.escape(title)}"></a>'
            '<div class="ft">'
            f'<div class="tt"><b>{_html.escape(title)}</b><span>{_html.escape(sub)}</span></div>'
            f'<a class="bt" href="{url}" target="_blank" rel="noopener">Xem ảnh ↗</a>'
            '</div></div>'
        )
    return '<div class="ag">' + "".join(cards) + '</div>'


res = ss.result
if res:
    p = {**res["payload"], "names": names}     # tên lấy từ ô nhập hiện tại
    mt = p["metrics"]

    block_title("target", "Kết quả phân tích")
    st.markdown(kpi_html([
        ("gauge", mt["max_shot_kmh"], "km/h", "Cú đánh nhanh nhất"),
        ("route", mt["distance_m"], "m", "Tổng quãng đường"),
        ("ball", mt["hits"], "", "Số pha chạm bóng"),
        ("target", f"{mt['ball_detect_rate'] * 100:.0f}", "%", "Tỉ lệ khung có bóng"),
    ]), unsafe_allow_html=True)

    if res.get("segmented"):
        r_cnt = len(p.get("rallies") or [])
        st.info(f"🎬 Video gồm {p.get('segment_count', r_cnt)} góc quay / pha bóng: "
                f"Hệ thống đã hợp nhất thành công {r_cnt} phán quyết và đồng bộ toàn bộ dòng thời gian vào Radar Hawk-Eye.")
    if res.get("warns"):
        st.markdown('<div class="meta" style="margin:0 0 14px">' +
                    "".join(chip(w, "warn", "alert") for w in res["warns"]) + '</div>', unsafe_allow_html=True)

    render_player(p, "/app/static/outputs/" + os.path.basename(res["mp4"]))
    if p.get("show", {}).get("referee", True):
        st.markdown(decision_html(p.get("decision"), p["names"], detected_doubles, rallies=p.get("rallies")), unsafe_allow_html=True)

    pngs = sorted(glob.glob(os.path.join(res["adir"], "*.png")))
    if pngs:
        block_title("image", "Biểu đồ phân tích")
        with st.expander(f"Xem {len(pngs)} biểu đồ (PNG)"):
            st.markdown(analysis_cards_html(pngs, res["session_id"]), unsafe_allow_html=True)
    with st.expander("Chi tiết đọc analytics"):
        st.json(res["dbg"])

    # ---------------------------------------------------------------- Lịch sử phiên
    block_title("history", "Lịch sử phiên")
    with card("history"):
        h1, h2, h3 = st.columns([1.3, 3, 1.3], vertical_alignment="center")
        if h1.button("Lưu vào lịch sử", disabled=res.get("saved", False), width='stretch'):
            try:
                if save_history(build_history_record(res)):
                    res["saved"] = True
                    st.rerun()
                else:
                    res["saved"] = True
                    h2.info("Phiên này đã có trong lịch sử.")
            except PermissionError:
                h2.error("Không ghi được `history/sessions.csv` (file đang mở trong Excel?). Đóng file rồi thử lại.")
            except Exception as e:  # noqa: BLE001
                h2.error(f"Lưu lịch sử thất bại: {e}")
        if res.get("saved"):
            h2.markdown(f'<div class="bar-st">{chip("Đã lưu", "ok", "check")}'
                        f'<span class="sub">Phiên <code>{_html.escape(res["session_id"])}</code> đã nằm trong lịch sử.</span></div>',
                        unsafe_allow_html=True)
        else:
            h2.markdown(f'<div class="bar-st">{chip("Chưa lưu", "warn", "dot")}'
                        f'<span class="sub">{_html.escape(res["video_name"])} · <code>{_html.escape(res["session_id"])}</code></span></div>',
                        unsafe_allow_html=True)
        h3.page_link("pages/1_dashboard.py", label="Mở Dashboard", width='stretch')

    # ---------------------------------------------------------------- Tải dữ liệu
    block_title("download", "Tải dữ liệu")
    d1, d2, d3, d4, d5 = st.columns(5)
    with open(res["mp4"], "rb") as f:
        d1.download_button("Video (.mp4)", f.read(), file_name=os.path.basename(res["mp4"]), width='stretch')
    with open(res["zip"], "rb") as f:
        d2.download_button("Analytics (.zip)", f.read(), file_name=os.path.basename(res["zip"]), width='stretch')
    if "log_path" in res and os.path.exists(res["log_path"]):
        with open(res["log_path"], "rb") as f:
            d3.download_button("Pipeline log (.txt)", f.read(), file_name=os.path.basename(res["log_path"]),
                               width='stretch')
    for col, key, label in ((d4, "summary", "Summary JSON"), (d5, "shots", "Shots CSV")):
        pth = res["found"].get(key)
        if pth and os.path.exists(pth):
            with open(pth, "rb") as f:
                col.download_button(label, f.read(), file_name=os.path.basename(pth), width='stretch')

    st.write("")
    if st.button("Tạo phiên mới", type="primary", width='stretch'):
        st.session_state.result = None
        st.rerun()