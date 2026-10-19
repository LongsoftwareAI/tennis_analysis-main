"""
Tennis Analysis – Dashboard tổng quan hệ thống
Đọc history/sessions.csv (do trang 2_tennis_analysis.py ghi khi bấm "Lưu vào lịch sử").
Giao diện dùng chung hệ thiết kế với trang Cài đặt / Phân tích.
"""
import html as _html
import os
import sys

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Dashboard - Tennis Analysis", page_icon="📊", layout="wide")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ui_common import ROOT, render_sidebar  # noqa: E402
render_sidebar()

HISTORY_PATH = os.path.join(ROOT, "history", "sessions.csv")
HISTORY_COLS = ["session_id", "saved_at", "video_name", "duration_sec", "max_shot_kmh", "distance_m", "hits",
                "decision", "point_winner", "nearest_line", "margin_cm", "landing_frame"]


def is_dark():
    try:
        return st.context.theme.type == "dark"
    except Exception:  # noqa: BLE001  (Streamlit cũ chưa có st.context.theme)
        return (st.get_option("theme.base") or "dark") == "dark"


DARK = is_dark()
TXT = "#e8ecf8" if DARK else "#1d2433"
GRID = "rgba(128,140,180,.24)"

# =====================================================================================
# Design tokens + CSS (cùng bộ với các trang khác)
# =====================================================================================
st.markdown("""
<style>
:root{--acc:#3b5bdb;--acc-2:#5c7cfa;--acc-soft:rgba(92,124,250,.16);--line:rgba(128,128,128,.26);
--surf:rgba(128,128,128,.07);--ok:#2fb344;--ok-soft:rgba(47,179,68,.15);--bad:#e5484d;--bad-soft:rgba(229,72,77,.15);
--warn:#f59f00;--warn-soft:rgba(245,159,0,.16);--t:.16s ease}
.block-container{padding-top:2.4rem;padding-bottom:4rem}
.pg-hd{display:flex;gap:16px;align-items:center}
.pg-ic{flex:none;width:52px;height:52px;border-radius:15px;display:grid;place-items:center;color:#fff;
background:linear-gradient(135deg,#182254 0%,#3b5bdb 100%);box-shadow:0 10px 24px rgba(59,91,219,.32)}
.pg-t{margin:0!important;padding:0!important;font-size:28px!important;font-weight:750!important;line-height:1.2!important;letter-spacing:-.01em}
.pg-s{margin:4px 0 0;opacity:.72;font-size:14px;line-height:1.55}
.meta{display:flex;flex-wrap:wrap;gap:8px;margin:18px 0 18px}
/* Chips */
.chip{display:inline-flex;align-items:center;gap:6px;padding:5px 11px;border-radius:999px;font-size:12.5px;font-weight:600;
line-height:1.2;border:1px solid var(--line);background:var(--surf);white-space:nowrap}
.chip.ok{background:var(--ok-soft);border-color:rgba(47,179,68,.45)}.chip.ok svg{color:var(--ok)}
.chip.bad{background:var(--bad-soft);border-color:rgba(229,72,77,.5)}.chip.bad svg{color:var(--bad)}
.chip.warn{background:var(--warn-soft);border-color:rgba(245,159,0,.5)}.chip.warn svg{color:var(--warn)}
.chip.acc{background:var(--acc-soft);border-color:rgba(92,124,250,.45)}.chip.acc svg{color:var(--acc-2)}
.chip svg{flex:none}
/* Card / section */
[class*="st-key-card_"]{border:1px solid var(--line)!important;border-radius:18px!important;background:var(--surf);padding:20px 22px!important}
.sec{display:flex;gap:12px;align-items:flex-start;margin:0 0 14px}
.sec-ic{flex:none;width:36px;height:36px;border-radius:10px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2)}
.sec-t{font-size:16px;font-weight:700;line-height:1.3}.sec-d{font-size:13px;opacity:.7;line-height:1.5;margin-top:2px}
.blk-t{display:flex;align-items:center;gap:10px;font-size:18px;font-weight:700;margin:30px 0 14px}
.blk-t svg{color:var(--acc-2)}
/* KPI */
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:14px;margin:0 0 6px}
.kpi{border:1px solid var(--line);border-radius:16px;background:var(--surf);padding:16px 18px;display:flex;gap:14px;align-items:center}
.kpi .ic{flex:none;width:42px;height:42px;border-radius:12px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2)}
.kpi .v{font-size:25px;font-weight:750;line-height:1.1;font-variant-numeric:tabular-nums}
.kpi .v small{font-size:13px;font-weight:600;opacity:.65;margin-left:4px}
.kpi .l{font-size:12.5px;opacity:.7;margin-top:3px}
/* Chart card header */
.cc-big{display:flex;align-items:baseline;flex-wrap:wrap;gap:6px 10px;margin:2px 0 2px}
.cc-big .v{font-size:34px;font-weight:800;line-height:1.1;font-variant-numeric:tabular-nums;letter-spacing:-.01em}
.cc-big .u{font-size:14px;font-weight:600;opacity:.65}
.cc-sub{font-size:12.5px;opacity:.65;margin-bottom:10px}
.cc-chips{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:6px}
.lgd{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:12px;opacity:.78;margin-top:2px}
.lgd span{display:inline-flex;align-items:center;gap:6px}
.lgd .sw{width:10px;height:10px;border-radius:3px;display:inline-block}
.lgd .dash{width:16px;height:0;border-top:2px dashed currentColor;opacity:.7;display:inline-block}
/* Table */
.tw{max-height:420px;overflow:auto;border:1px solid var(--line);border-radius:14px}
.tb{width:100%;border-collapse:separate;border-spacing:0;font-size:14px;font-variant-numeric:tabular-nums}
.tb th{position:sticky;top:0;z-index:1;background:var(--th);text-align:left;font-size:12px;letter-spacing:.03em;font-weight:700;padding:11px 14px;white-space:nowrap}
.tb td{padding:12px 14px;border-top:1px solid var(--line);vertical-align:middle}
.tb tr:hover td{background:var(--acc-soft)}
.tb td.num,.tb th.num{text-align:right}
.tb .mut{display:block;font-size:12px;opacity:.62;margin-top:2px}
.tb code{font-size:11.5px;padding:1px 5px;border-radius:5px;background:var(--surf)}
/* Widgets */
[data-testid="stExpander"]{border:1px solid var(--line)!important;border-radius:14px!important;background:var(--surf)}
[data-testid="stExpander"] summary{font-weight:600}
[data-testid="stRadio"] div[role="radiogroup"]{gap:4px;background:var(--surf);padding:4px;border-radius:12px;width:fit-content;max-width:100%;flex-wrap:wrap;border:1px solid var(--line)}
[data-testid="stRadio"] label{margin:0!important;padding:7px 16px;border-radius:9px;cursor:pointer;transition:background var(--t),color var(--t)}
[data-testid="stRadio"] label>div:first-child{display:none}
[data-testid="stRadio"] label:hover{background:var(--acc-soft)}
[data-testid="stRadio"] label:has(input:checked){background:var(--acc);color:#fff}
[data-testid="stRadio"] label:has(input:checked) *{color:#fff!important}
[data-testid="stRadio"] label:has(input:focus-visible){outline:2px solid var(--acc-2);outline-offset:2px}
div.stButton>button,div.stDownloadButton>button{min-height:46px;border-radius:12px;font-weight:600;transition:background var(--t),border-color var(--t)}
div.stButton>button:focus-visible,div.stDownloadButton>button:focus-visible{outline:2px solid var(--acc-2);outline-offset:2px}
.empty{display:flex;flex-direction:column;align-items:center;text-align:center;gap:10px;padding:34px 10px}
.empty .ic{width:64px;height:64px;border-radius:18px;display:grid;place-items:center;background:var(--acc-soft);color:var(--acc-2)}
.empty b{font-size:18px}.empty span{opacity:.7;font-size:14px;max-width:460px;line-height:1.55}
@media (prefers-reduced-motion:reduce){*{transition:none!important}}
</style>
""", unsafe_allow_html=True)
st.markdown(f"<style>:root{{--th:{'#1c2133' if DARK else '#eceff7'}}}</style>", unsafe_allow_html=True)

# =====================================================================================
# Icon + helper HTML
# =====================================================================================
_ICONS = {
    "bars": '<line x1="6" y1="20" x2="6" y2="12"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="18" y1="20" x2="18" y2="9"/>',
    "up": '<path d="M12 19V5M6 11l6-6 6 6"/>',
    "down": '<path d="M12 5v14M6 13l6 6 6-6"/>',
    "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
    "film": '<rect x="3" y="4" width="18" height="16" rx="2"/><path d="M7 4v16M17 4v16M3 9h4M3 15h4M17 9h4M17 15h4"/>',
    "gauge": '<path d="M4 17a8 8 0 1 1 16 0"/><path d="M12 17l4-5"/>',
    "route": '<circle cx="6" cy="18" r="2"/><circle cx="18" cy="6" r="2"/><path d="M8 18h6a3 3 0 0 0 0-6h-4a3 3 0 0 1 0-6h6"/>',
    "ball": '<circle cx="12" cy="12" r="9"/><path d="M5 6c4 3 4 9 0 12M19 6c-4 3-4 9 0 12"/>',
    "target": '<circle cx="12" cy="12" r="9"/><circle cx="12" cy="12" r="5"/><circle cx="12" cy="12" r="1" fill="currentColor"/>',
    "eye": '<path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z"/><circle cx="12" cy="12" r="3"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7"/><path d="M3 4v5h5"/><path d="M12 8v4l3 2"/>',
    "check": '<circle cx="12" cy="12" r="9"/><path d="M8 12.5l2.7 2.7L16 9.5"/>',
    "alert": '<path d="M12 3l10 18H2L12 3z"/><line x1="12" y1="10" x2="12" y2="15"/><line x1="12" y1="18" x2="12" y2="18.1"/>',
    "dot": '<circle cx="12" cy="12" r="4" fill="currentColor"/>',
    "inbox": '<path d="M3 13l3-8h12l3 8v6H3v-6z"/><path d="M3 13h5l1 3h6l1-3h5"/>',
    "list": '<line x1="8" y1="6" x2="20" y2="6"/><line x1="8" y1="12" x2="20" y2="12"/><line x1="8" y1="18" x2="20" y2="18"/><circle cx="4" cy="6" r="1" fill="currentColor"/><circle cx="4" cy="12" r="1" fill="currentColor"/><circle cx="4" cy="18" r="1" fill="currentColor"/>',
}


def svg(name, size=18):
    return (f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" '
            f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{_ICONS[name]}</svg>')


def chip(text, kind="", icon=None):
    ic = svg(icon, 14) if icon else ""
    return f'<span class="chip {kind}">{ic}{_html.escape(str(text))}</span>'


def sec_html(icon, title, desc=""):
    return (f'<div class="sec"><div class="sec-ic">{svg(icon, 18)}</div><div>'
            f'<div class="sec-t" role="heading" aria-level="2">{title}</div><div class="sec-d">{desc}</div></div></div>')


def block_title(icon, text):
    st.markdown(f'<div class="blk-t" role="heading" aria-level="2">{svg(icon, 20)}{text}</div>', unsafe_allow_html=True)


def card(name):
    try:
        return st.container(border=True, key=f"card_{name}")
    except TypeError:
        return st.container(border=True)


def kpi_html(items):
    cells = "".join(
        f'<div class="kpi"><div class="ic">{svg(ic, 22)}</div><div><div class="v">{v}<small>{u}</small></div>'
        f'<div class="l">{l}</div></div></div>' for ic, v, u, l in items)
    return f'<div class="kpis">{cells}</div>'


def fmt_duration(sec):
    sec = int(round(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"


# =====================================================================================
# Biểu đồ phiên (Altair): cột / vùng+đường, đường trung bình, nhãn giá trị, tooltip đầy đủ
# =====================================================================================
def session_chart(d, col, color, kind, unit_title, fmt):
    n = len(d)
    avg = float(d[col].mean())
    order = list(d["label"])
    x = alt.X("label:N", sort=order, axis=alt.Axis(title=None, labelAngle=0, labelColor=TXT, labelOverlap=True,
                                                    ticks=False, domain=False, labelFontSize=11))
    y = alt.Y(f"{col}:Q", scale=alt.Scale(domainMin=0, nice=True),
              axis=alt.Axis(title=None, labelColor=TXT, gridColor=GRID, tickCount=4, domain=False, ticks=False,
                            labelFontSize=11))
    tip = [alt.Tooltip("session_id:N", title="Phiên"), alt.Tooltip("video_name:N", title="Video"),
           alt.Tooltip("saved_at:N", title="Lưu lúc"), alt.Tooltip(f"{col}:Q", title=unit_title, format=fmt)]
    rule = alt.Chart(pd.DataFrame({"v": [avg]})).mark_rule(strokeDash=[5, 4], color=TXT, opacity=.55, strokeWidth=1.4
                                                           ).encode(y="v:Q")
    if kind == "bar":
        size = 28 if n <= 10 else (16 if n <= 20 else 9)
        main = alt.Chart(d).mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6, size=size, color=color).encode(
            x=x, y=y, opacity=alt.condition("datum.is_best", alt.value(1), alt.value(.5)), tooltip=tip)
    else:
        area = alt.Chart(d).mark_area(color=color, opacity=.16, interpolate="monotone").encode(x=x, y=y)
        line = alt.Chart(d).mark_line(color=color, strokeWidth=2.6, interpolate="monotone").encode(x=x, y=y)
        pts = alt.Chart(d).mark_point(filled=True, color=color, opacity=1, stroke="white", strokeWidth=1.2).encode(
            x=x, y=y, size=alt.condition("datum.is_best", alt.value(150), alt.value(70)), tooltip=tip)
        main = area + line + pts
    layers = main + rule
    if n <= 12:
        layers = layers + alt.Chart(d).mark_text(dy=-11, fontSize=11, fontWeight="bold", color=TXT).encode(
            x=x, y=y, text=alt.Text(f"{col}:Q", format=fmt))
    return layers.properties(height=230).configure_view(strokeWidth=0).configure(background="transparent")


def legend_html(color, kind):
    best = ("Phiên cao nhất" if kind == "bar" else "Phiên cao nhất (điểm lớn)")
    other = "Các phiên khác" if kind == "bar" else "Xu hướng"
    sw2 = (f'<span><i class="sw" style="background:{color};opacity:.5"></i>{other}</span>')
    return (f'<div class="lgd"><span><i class="sw" style="background:{color}"></i>{best}</span>{sw2}'
            f'<span><i class="dash"></i>Trung bình</span></div>')


def chart_card(col_ui, key, icon, title, desc, d, col, unit, color, kind, fmt, extra):
    """extra = ("Thấp nhất"|"Tổng", giá trị)."""
    v_last = float(d[col].iloc[-1])
    delta = float(d[col].iloc[-1] - d[col].iloc[-2]) if len(d) >= 2 else None
    if delta is None:
        dchip = chip("Phiên đầu tiên", "acc", "dot")
    elif abs(delta) < 1e-9:
        dchip = chip("Không đổi so với phiên trước", "", "dot")
    else:
        dchip = chip(f"{delta:+,.1f} so với phiên trước", "acc", "up" if delta > 0 else "down")
    best = float(d[col].max())
    chips = (chip(f"Cao nhất: {best:,.1f}", "ok", "check") + chip(f"Trung bình: {d[col].mean():,.1f}", "acc", "bars") +
             chip(f"{extra[0]}: {extra[1]}", "", "dot"))
    with col_ui, card(key):
        st.markdown(sec_html(icon, title, desc) +
                    f'<div class="cc-big"><span class="v">{v_last:,.{0 if fmt == ".0f" else 1}f}</span>'
                    f'<span class="u">{unit}</span></div>'
                    f'<div class="cc-sub">Phiên mới nhất · {_html.escape(str(d["label"].iloc[-1]))}</div>'
                    f'<div class="cc-chips">{dchip}</div><div class="cc-chips">{chips}</div>', unsafe_allow_html=True)
        st.altair_chart(session_chart(d, col, color, kind, f"{title} ({unit})", ",.0f" if fmt == ".0f" else ",.1f"),
                        width='stretch')
        st.markdown(legend_html(color, kind), unsafe_allow_html=True)


# =====================================================================================
# Bảng phán quyết Hawk-Eye (HTML, chip IN / OUT / NET)
# =====================================================================================
def decision_chip(v):
    v = str(v or "").strip().upper()
    if v in ("IN", "WINNER"):
        return chip(v, "ok", "check")
    if v == "OUT":
        return chip(v, "bad", "alert")
    if v == "NET":
        return chip(v, "warn", "alert")
    return chip(v or "-", "", "dot")


def esc(v, dash="-"):
    return dash if (v is None or pd.isna(v) or str(v) == "") else _html.escape(str(v))


def hawk_table(df):
    rows = []
    for _, r in df.iloc[::-1].iterrows():                       # phiên mới nhất lên đầu
        margin = f"{r['margin_cm']:+.1f}" if pd.notna(r["margin_cm"]) else "-"
        frame = f"{int(r['landing_frame'])}" if pd.notna(pd.to_numeric(r["landing_frame"], errors="coerce")) else "-"
        rows.append(
            f'<tr><td><b>{esc(r["video_name"])}</b><span class="mut">{esc(r["saved_at"])} · <code>{esc(r["session_id"])}</code></span></td>'
            f'<td>{decision_chip(r["decision"])}</td><td>{esc(r["point_winner"])}</td><td>{esc(r["nearest_line"])}</td>'
            f'<td class="num">{margin}</td><td class="num">{frame}</td></tr>')
    head = ("<tr><th>Phiên phân tích</th><th>Quyết định cuối cùng</th><th>Điểm cho</th><th>Vạch gần nhất</th>"
            '<th class="num">Margin (cm)</th><th class="num">Bóng chạm đất lần 1 (frame)</th></tr>')
    return f'<div class="tw"><table class="tb">{head}{"".join(rows)}</table></div>'


# =====================================================================================
# Header
# =====================================================================================
h_l, h_r = st.columns([4, 1.4], vertical_alignment="center")
h_l.markdown(
    f'<div class="pg-hd"><div class="pg-ic">{svg("bars", 26)}</div><div>'
    '<h1 class="pg-t">Dashboard tổng quan</h1>'
    '<p class="pg-s">Thống kê các phiên đã lưu: tốc độ cú đánh, quãng đường, số pha chạm bóng và phán quyết Hawk-Eye.</p>'
    '</div></div>', unsafe_allow_html=True)
h_r.page_link("pages/2_tennis_analysis.py", label="Phân tích video", width='stretch')

# =====================================================================================
# Trạng thái rỗng
# =====================================================================================
def empty_state(msg):
    st.write("")
    with card("empty"):
        st.markdown(f'<div class="empty"><div class="ic">{svg("inbox", 30)}</div><b>Chưa có dữ liệu</b><span>{msg}</span></div>',
                    unsafe_allow_html=True)
        c = st.columns([1, 1, 1])[1]
        c.page_link("pages/2_tennis_analysis.py", label="Đến trang phân tích", width='stretch')
    st.stop()


if not os.path.exists(HISTORY_PATH):
    empty_state('Hãy phân tích một video ở trang Phân tích rồi bấm "Lưu vào lịch sử" để phiên xuất hiện tại đây.')

df = pd.read_csv(HISTORY_PATH)
if df.empty:
    empty_state("Lịch sử đang trống. Lưu ít nhất một phiên phân tích để xem thống kê.")

for c in HISTORY_COLS:
    if c not in df.columns:
        df[c] = np.nan
for c in ("duration_sec", "max_shot_kmh", "distance_m", "hits", "margin_cm"):
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.sort_values("saved_at").reset_index(drop=True)
df["no"] = np.arange(1, len(df) + 1)
df["label"] = "#" + df["no"].astype(str)
for c in ("session_id", "video_name", "saved_at"):
    df[c] = df[c].astype(str)

st.markdown('<div class="meta">' + chip(f"{len(df)} phiên đã lưu", "acc", "history") +
            chip(f"Cập nhật gần nhất: {df['saved_at'].iloc[-1]}", "", "clock") + '</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------ Hàng 1: KPI tổng
best_all = df["max_shot_kmh"].max()
st.markdown(kpi_html([
    ("history", len(df), "", "Tổng số phiên xử lý"),
    ("film", df["video_name"].nunique(), "", "Số video khác nhau"),
    ("clock", fmt_duration(df["duration_sec"].sum()), "", "Tổng thời gian các phiên"),
    ("gauge", f"{best_all:.1f}" if pd.notna(best_all) else "-", "km/h", "Cú đánh nhanh nhất mọi phiên"),
]), unsafe_allow_html=True)

# ------------------------------------------------------------------ Hàng 2: 3 biểu đồ
block_title("bars", "Xu hướng theo phiên")
RANGES = {"5 phiên gần nhất": 5, "10 phiên gần nhất": 10, "20 phiên gần nhất": 20, "Tất cả": None}
rng = st.radio("Phạm vi hiển thị", list(RANGES), index=1, horizontal=True, key="dash_range", label_visibility="collapsed")
n_show = RANGES[rng]
view = df.tail(n_show) if n_show else df


def prep(col):
    d = view.dropna(subset=[col]).copy()
    if not d.empty:
        d["is_best"] = d[col] == d[col].max()
    return d


c1, c2, c3 = st.columns(3, gap="medium")
specs = [
    (c1, "shot", "gauge", "Cú đánh nhanh nhất", "Tốc độ cú đánh cao nhất mỗi phiên", "max_shot_kmh", "km/h", "#5c7cfa", "bar", ".1f", "Thấp nhất"),
    (c2, "dist", "route", "Tổng quãng đường", "Quãng đường các cầu thủ di chuyển mỗi phiên", "distance_m", "m", "#2fb344", "line", ".1f", "Tổng"),
    (c3, "hits", "ball", "Số pha chạm bóng", "Số lần bóng được đánh trong mỗi phiên", "hits", "pha", "#f59f00", "bar", ".0f", "Tổng"),
]
for ui, key, icon, title, desc, col, unit, color, kind, fmt, ex in specs:
    d = prep(col)
    if d.empty:
        with ui, card(f"c_{key}"):
            st.markdown(sec_html(icon, title, desc) + chip("Chưa có dữ liệu cho chỉ số này", "warn", "alert"),
                        unsafe_allow_html=True)
        continue
    if ex == "Thấp nhất":
        extra = (ex, f"{d[col].min():,.1f}")
    else:
        extra = (ex, f"{d[col].sum():,.0f}" if fmt == ".0f" else f"{d[col].sum():,.1f} {unit}")
    chart_card(ui, f"c_{key}", icon, title, desc, d, col, unit, color, kind, fmt, extra)
st.caption(f"Biểu đồ hiển thị {len(view)}/{len(df)} phiên. Rê chuột vào cột hoặc điểm để xem video, thời điểm lưu và giá trị chính xác.")

# ------------------------------------------------------------------ Hàng 3: bảng Hawk-Eye
block_title("eye", "Phán quyết của Trọng tài Hawk-Eye")
dec = df["decision"].astype(str).str.strip().str.upper()
n_in = int(dec.isin(["IN", "WINNER"]).sum())
n_out = int((dec == "OUT").sum())
n_net = int((dec == "NET").sum())
n_oth = len(df) - n_in - n_out - n_net
with card("hawk"):
    st.markdown('<div class="meta" style="margin:0 0 14px">' + chip(f"IN: {n_in}", "ok", "check") +
                chip(f"OUT: {n_out}", "bad", "alert") + chip(f"NET: {n_net}", "warn", "alert") +
                (chip(f"Khác: {n_oth}", "", "dot") if n_oth else "") + '</div>' + hawk_table(df), unsafe_allow_html=True)

st.write("")
with st.expander("Dữ liệu thô (sessions.csv)"):
    st.dataframe(df.drop(columns=["no", "label"]), hide_index=True, width='stretch')
    with open(HISTORY_PATH, "rb") as f:
        st.download_button("Tải sessions.csv", f.read(), file_name="sessions.csv")