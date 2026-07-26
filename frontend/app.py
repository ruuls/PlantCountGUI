import os
import re
import html
import urllib.parse
import io
import base64
import hashlib
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import requests
import httpx
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import streamlit as st

st.set_page_config(
    page_title="PlantCountGUI · Digital Ag Technologies Lab",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_FONT = int(os.getenv("BASE_FONT_PT", "15"))
SHOW_DEBUG = bool(int(os.getenv("UI_DEBUG", "0")))
BOLD_FAMILY = os.getenv("BOLD_FONT_FAMILY", "IBM Plex Sans, Segoe UI, sans-serif")
BAR_TEXT_MULT = float(os.getenv("BAR_TEXT_MULT", "1.1"))

# ---------- design tokens ----------
BG = "#080B0A"
PANEL = "#111614"
PANEL_SOFT = "#161D1A"
BORDER = "rgba(233, 242, 236, 0.10)"
BORDER_STRONG = "rgba(233, 242, 236, 0.18)"
INK = "#E9EFEA"
INK_MUTED = "#93A29A"
ACCENT = "#E5751F"        # VT burnt orange
ACCENT_DEEP = "#861F41"   # VT chicago maroon
OK = "#3FBF7F"

# High-contrast on both the dark chrome and on aerial imagery.
PALETTE = ["#FF8A3D", "#3FD0C9", "#8FD14F", "#F2C14E", "#9B8CFF", "#FF6B9A", "#5BA8FF"]

PLOT_INK = INK
PLOT_GRID = "rgba(233, 242, 236, 0.08)"


def bump_bar_label_fonts(fig, base=BASE_FONT):
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
        insidetextfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
        outsidetextfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
    )
    fig.update_layout(uniformtext_minsize=int(base * BAR_TEXT_MULT), uniformtext_mode="show")
    return fig


def apply_plotly_fonts(fig, base=BASE_FONT):
    fig.update_layout(
        font=dict(size=base, color=PLOT_INK, family=BOLD_FAMILY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(size=int(base * 0.9), color=PLOT_INK),
            bgcolor="rgba(8, 11, 10, 0.72)",
            bordercolor="rgba(233, 242, 236, 0.14)",
            borderwidth=1,
        ),
        hoverlabel=dict(
            font_size=int(base * 0.9),
            font_family=BOLD_FAMILY,
            font_color=PLOT_INK,
            bgcolor="#12201a",
            bordercolor="rgba(233, 242, 236, 0.25)",
        ),
    )
    fig.update_xaxes(
        title_font=dict(size=int(base * 1.0), color=INK_MUTED, family=BOLD_FAMILY),
        tickfont=dict(size=int(base * 0.9), color=PLOT_INK, family=BOLD_FAMILY),
        gridcolor=PLOT_GRID,
        zerolinecolor=PLOT_GRID,
    )
    fig.update_yaxes(
        title_font=dict(size=int(base * 1.0), color=INK_MUTED, family=BOLD_FAMILY),
        tickfont=dict(size=int(base * 0.9), color=PLOT_INK, family=BOLD_FAMILY),
        gridcolor=PLOT_GRID,
        zerolinecolor=PLOT_GRID,
    )
    return fig


# Styling is intentionally limited to color / border / typography rules on
# existing Streamlit nodes. No overlays, no positioned pseudo-elements — those
# previously swallowed clicks.
def inject_theme_css():
    st.markdown(
        f"""
<style>
[data-testid="stAppViewContainer"] {{ background: {BG}; }}
[data-testid="stHeader"] {{ background: transparent; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1480px; }}

h1, h2, h3, h4 {{ letter-spacing: -0.01em; }}

/* ---------- masthead: logo on top, title below ---------- */
.pui-hero {{
    padding: 1.35rem 1.5rem 1.4rem; margin-bottom: 1.6rem;
    border: 1px solid {BORDER}; border-top: 3px solid {ACCENT_DEEP}; border-radius: 10px;
    background: {PANEL};
}}
.pui-hero-top {{
    display: flex; align-items: center; justify-content: space-between;
    gap: 1rem; flex-wrap: wrap; margin-bottom: 1rem;
}}
.pui-hero-logo {{
    background: #fff; border-radius: 8px; padding: 9px 12px; line-height: 0;
}}
.pui-hero-logo img {{ height: 62px; width: auto; display: block; }}
.pui-hero h1 {{
    font-size: 2.3rem; font-weight: 700; margin: 0; padding: 0; color: {INK}; line-height: 1.12;
}}
.pui-hero p {{ margin: 0.35rem 0 0; color: {INK_MUTED}; font-size: 0.95rem; }}
.pui-eyebrow {{
    font-size: 0.72rem; letter-spacing: 0.16em; text-transform: uppercase;
    color: {INK_MUTED}; font-weight: 700; margin-bottom: 0.45rem;
}}
.pui-hero-rule {{ height: 1px; background: {BORDER}; margin: 1.05rem 0 0.95rem; }}
.pui-pill {{
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.32rem 0.75rem; border-radius: 4px; font-size: 0.78rem; font-weight: 600;
    border: 1px solid rgba(63, 191, 127, 0.30); background: rgba(63, 191, 127, 0.10); color: {OK};
    white-space: nowrap;
}}
.pui-pill .dot {{
    width: 8px; height: 8px; border-radius: 50%; background: {OK};
    box-shadow: 0 0 0 3px rgba(63,191,127,0.18);
}}

/* ---------- section headers ---------- */
.pui-section {{ display: flex; align-items: flex-start; gap: 0.7rem; margin: 1.9rem 0 0.85rem; }}
.pui-step {{
    flex: 0 0 auto; width: 24px; height: 24px; border-radius: 4px;
    display: inline-flex; align-items: center; justify-content: center;
    font-size: 0.78rem; font-weight: 700; color: {ACCENT};
    background: rgba(229,117,31,0.12); border: 1px solid rgba(229,117,31,0.30);
    margin-top: 3px;
}}
.pui-section h2 {{
    margin: 0 !important; padding: 0 !important;
    font-size: 1.14rem; font-weight: 700; color: {INK}; line-height: 1.3;
}}
.pui-section p {{ margin: 0.12rem 0 0 !important; font-size: 0.88rem; color: {INK_MUTED}; }}

/* ---------- KPI cards ---------- */
.pui-kpi-row {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 0.25rem 0 0.4rem; }}
.pui-kpi {{
    flex: 1 1 200px; min-width: 190px;
    padding: 0.9rem 1.05rem 0.95rem; border-radius: 8px;
    background: {PANEL}; border: 1px solid {BORDER};
    border-left: 3px solid var(--kpi-accent, {ACCENT});
}}
.pui-kpi-label {{
    font-size: 0.72rem; letter-spacing: 0.09em; text-transform: uppercase;
    color: {INK_MUTED}; font-weight: 700;
}}
.pui-kpi-value {{
    font-size: 1.85rem; font-weight: 700; color: {INK}; line-height: 1.15; margin-top: 0.3rem;
    overflow-wrap: anywhere;
}}
.pui-kpi-value .unit {{ font-size: 0.92rem; font-weight: 600; color: {INK_MUTED}; margin-left: 0.2rem; }}
.pui-kpi-sub {{ font-size: 0.79rem; color: {INK_MUTED}; margin-top: 0.3rem; }}

/* ---------- class chips ---------- */
.pui-chips {{ display: flex; gap: 0.45rem; flex-wrap: wrap; margin: 0.7rem 0 0.2rem; }}
.pui-chip {{
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.26rem 0.65rem; border-radius: 4px;
    font-size: 0.83rem; font-weight: 600; color: {INK};
    background: rgba(233,242,236,0.04); border: 1px solid {BORDER_STRONG};
}}
.pui-chip .swatch {{ width: 9px; height: 9px; border-radius: 2px; }}
.pui-chip .n {{ color: {INK_MUTED}; font-weight: 600; }}

/* ---------- how-it-works cards ---------- */
.pui-steps {{ display: flex; gap: 0.75rem; flex-wrap: wrap; margin-top: 0.4rem; }}
.pui-card {{
    flex: 1 1 220px; padding: 1rem 1.1rem; border-radius: 8px;
    background: {PANEL}; border: 1px solid {BORDER};
}}
.pui-card .n {{
    font-size: 0.71rem; font-weight: 700; letter-spacing: 0.12em; color: {ACCENT};
    text-transform: uppercase;
}}
.pui-card h4 {{ margin: 0.35rem 0 0.3rem; font-size: 0.97rem; color: {INK}; font-weight: 700; }}
.pui-card p {{ margin: 0; font-size: 0.86rem; color: {INK_MUTED}; line-height: 1.5; }}

/* ---------- bordered containers as panels ---------- */
[data-testid="stVerticalBlockBorderWrapper"] {{
    border-radius: 8px !important;
    border-color: {BORDER} !important;
    background: {PANEL};
}}

/* ---------- buttons ---------- */
[data-testid="stBaseButton-primary"], button[kind="primary"] {{
    background: {ACCENT} !important;
    border: 1px solid {ACCENT} !important;
    color: #150A02 !important;
    font-weight: 700 !important; letter-spacing: 0.01em;
    border-radius: 6px !important; padding: 0.55rem 1rem !important;
}}
[data-testid="stBaseButton-primary"]:hover, button[kind="primary"]:hover {{
    background: #cf6316 !important; border-color: #cf6316 !important; color: #150A02 !important;
}}
[data-testid="stBaseButton-primary"]:disabled, button[kind="primary"]:disabled {{
    background: {PANEL_SOFT} !important; color: {INK_MUTED} !important;
    border: 1px solid {BORDER} !important;
}}
[data-testid="stBaseButton-secondary"], [data-testid="stDownloadButton"] button, button[kind="secondary"] {{
    background: {PANEL_SOFT} !important; color: {INK} !important;
    border: 1px solid {BORDER_STRONG} !important; border-radius: 6px !important;
    font-weight: 600 !important;
}}
[data-testid="stBaseButton-secondary"]:hover, [data-testid="stDownloadButton"] button:hover {{
    border-color: {ACCENT} !important; color: {ACCENT} !important;
}}

/* ---------- inputs ---------- */
[data-testid="stFileUploaderDropzone"] {{
    background: rgba(233, 242, 236, 0.02) !important;
    border: 1px dashed {BORDER_STRONG} !important; border-radius: 8px !important;
}}
[data-testid="stFileUploaderDropzone"]:hover {{ border-color: {ACCENT} !important; }}
[data-testid="stFileUploaderFile"] {{
    background: rgba(63, 191, 127, 0.06); border: 1px solid rgba(63,191,127,0.20);
    border-radius: 6px; padding: 0.35rem 0.6rem;
}}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {{
    background: {PANEL_SOFT} !important; border-color: {BORDER_STRONG} !important;
    border-radius: 6px !important;
}}
/* multiselect tags were white-on-white with the old primary color */
span[data-baseweb="tag"] {{
    background: rgba(229, 117, 31, 0.16) !important;
    border: 1px solid rgba(229, 117, 31, 0.40) !important;
    color: {INK} !important; border-radius: 4px !important;
}}
span[data-baseweb="tag"] span, span[data-baseweb="tag"] svg {{ color: {INK} !important; fill: {INK} !important; }}
label p, [data-testid="stWidgetLabel"] p {{
    font-size: 0.85rem !important; font-weight: 600 !important; color: {INK_MUTED} !important;
}}

/* ---------- tabs ---------- */
[data-baseweb="tab-list"] {{
    gap: 0.35rem; border-bottom: 1px solid {BORDER};
    background: transparent; padding-bottom: 0;
}}
[data-baseweb="tab"] {{
    background: transparent !important; border-radius: 4px 4px 0 0;
    padding: 0.55rem 0.95rem !important; color: {INK_MUTED} !important; font-weight: 600;
}}
[data-baseweb="tab"][aria-selected="true"] {{
    color: {INK} !important; background: rgba(229,117,31,0.08) !important;
}}
[data-baseweb="tab-highlight"] {{ background: {ACCENT} !important; height: 2px; }}
[data-baseweb="tab-border"] {{ background: transparent !important; }}

/* ---------- misc chrome ---------- */
[data-testid="stAlert"] {{ border-radius: 8px; border: 1px solid {BORDER}; }}
[data-testid="stExpander"] details {{
    background: {PANEL} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important;
}}
[data-testid="stMetricValue"] {{ font-weight: 700; }}
hr {{ border-color: {BORDER}; }}
[data-testid="stSidebar"] {{
    background: #0C110F;
    border-right: 1px solid {BORDER};
}}
[data-testid="stSidebar"] .pui-side-title {{
    font-size: 0.78rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: {ACCENT}; font-weight: 700; margin: 0.2rem 0 0.7rem;
}}
.pui-side-step {{
    display: flex; gap: 0.6rem; align-items: baseline;
    font-size: 0.95rem; color: {INK}; margin-bottom: 0.62rem; line-height: 1.4;
}}
.pui-side-step .idx {{
    flex: 0 0 auto; font-size: 0.78rem; font-weight: 700; color: {ACCENT};
    background: rgba(229,117,31,0.12); border-radius: 3px; padding: 0.08rem 0.4rem;
}}
.pui-meta {{
    font-size: 0.86rem; color: {INK_MUTED}; line-height: 1.55;
    background: {PANEL}; border: 1px solid {BORDER}; border-radius: 6px; padding: 0.6rem 0.75rem;
    overflow-wrap: anywhere;
}}
.pui-foot {{
    margin-top: 2.6rem; padding-top: 1rem; border-top: 1px solid {BORDER};
    font-size: 0.79rem; color: {INK_MUTED};
}}
.pui-hint {{ font-size: 0.84rem; color: {INK_MUTED}; margin: 0.2rem 0 0.6rem; }}
</style>
""",
        unsafe_allow_html=True,
    )


inject_theme_css()


# ---------- small render helpers ----------
def section(title: str, subtitle: str = "", step: str | None = None):
    badge = f'<span class="pui-step">{html.escape(step)}</span>' if step else ""
    sub = f'<p>{html.escape(subtitle)}</p>' if subtitle else ""
    st.markdown(
        f'<div class="pui-section">{badge}<div><h2>{html.escape(title)}</h2>{sub}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(cards: list[dict]):
    blocks = []
    for c in cards:
        unit = f'<span class="unit">{html.escape(c["unit"])}</span>' if c.get("unit") else ""
        sub = html.escape(c.get("sub", ""))
        blocks.append(
            f'<div class="pui-kpi" style="--kpi-accent:{c.get("accent", ACCENT)}">'
            f'<div class="pui-kpi-label">{html.escape(c["label"])}</div>'
            f'<div class="pui-kpi-value">{html.escape(str(c["value"]))}{unit}</div>'
            f'<div class="pui-kpi-sub">{sub}</div>'
            f"</div>"
        )
    st.markdown(f'<div class="pui-kpi-row">{"".join(blocks)}</div>', unsafe_allow_html=True)


def class_chips(names: list[str], counts: list[int]):
    chips = []
    for i, name in enumerate(names):
        color = PALETTE[i % len(PALETTE)]
        chips.append(
            f'<span class="pui-chip"><span class="swatch" style="background:{color}"></span>'
            f'{html.escape(name)}<span class="n">{int(counts[i]):,}</span></span>'
        )
    if chips:
        st.markdown(f'<div class="pui-chips">{"".join(chips)}</div>', unsafe_allow_html=True)


def how_it_works():
    steps = [
        ("Step 1", "Upload an orthomosaic", "A georeferenced GeoTIFF (.tif/.tiff). Coordinates are read from the raster, so no manual alignment is needed."),
        ("Step 2", "Pick a data source", "Run the YOLO detector over image tiles, optionally deduplicating overlapping hits — or plot detections you already have in a CSV."),
        ("Step 3", "Review and export", "Inspect detections on the map, study density hotspots, then export CSV or per-class shapefiles for GIS."),
    ]
    cards = "".join(
        f'<div class="pui-card"><div class="n">{n}</div><h4>{html.escape(t)}</h4><p>{html.escape(d)}</p></div>'
        for n, t, d in steps
    )
    st.markdown(f'<div class="pui-steps">{cards}</div>', unsafe_allow_html=True)


PLOTLY_CONFIG_STATIC = {"displayModeBar": False, "displaylogo": False}
PLOTLY_CONFIG_MAP = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToRemove": ["select2d", "lasso2d", "autoScale2d"],
}


# ---------- helpers ----------
def _clean(s: str) -> str:
    s = "".join(ch for ch in s if ord(ch) >= 32)
    return s.strip().rstrip("/")


def resolve_api_base() -> str:
    candidate = os.getenv("API_BASE") or "http://127.0.0.1:8000"
    candidate = _clean(candidate)
    if not re.match(r"^https?://", candidate):
        candidate = "http://" + candidate
    candidate = candidate.replace("0.0.0.0", "127.0.0.1")
    parsed = urllib.parse.urlparse(candidate)
    if not parsed.scheme or not parsed.netloc or " " in candidate:
        st.error("Backend URL is invalid. Set API_BASE like http://127.0.0.1:8000")
        st.stop()
    return candidate


API_BASE = resolve_api_base()


def sha1_bytes(b: bytes) -> str:
    h = hashlib.sha1()
    h.update(b)
    return h.hexdigest()


def sanitize_filename(name: str | None) -> str:
    if not name:
        return "image.tif"
    return re.sub(r"[^A-Za-z0-9._\-]+", "_", name) or "image.tif"


# ---------- HTTP clients ----------
requests_sess = requests.Session()
requests_sess.trust_env = False

httpx_client = httpx.Client(
    timeout=httpx.Timeout(connect=15.0, read=600.0, write=600.0, pool=15.0),
    follow_redirects=False,
    transport=httpx.HTTPTransport(retries=0),
    trust_env=False,
    http2=False,
)


# ---------- header ----------
_HEADER_PATH = Path(__file__).resolve().parent / "assets" / "vt_digital_ag_header.png"


def render_hero(backend_ok: bool):
    logo = ""
    if _HEADER_PATH.is_file():
        _b64 = base64.b64encode(_HEADER_PATH.read_bytes()).decode("ascii")
        logo = (
            f'<div class="pui-hero-logo">'
            f'<img src="data:image/png;base64,{_b64}" alt="Virginia Tech Digital Ag Technologies Lab" />'
            f"</div>"
        )
    if backend_ok:
        status = '<span class="pui-pill"><span class="dot"></span>Detection service online</span>'
    else:
        status = (
            '<span class="pui-pill" style="border-color:rgba(255,107,154,0.4);'
            'background:rgba(255,107,154,0.12);color:#FF6B9A">'
            '<span class="dot" style="background:#FF6B9A;box-shadow:0 0 0 3px rgba(255,107,154,0.18)"></span>'
            "Detection service offline</span>"
        )
    st.markdown(
        f'<div class="pui-hero">'
        f'<div class="pui-hero-top">{logo}{status}</div>'
        f'<div class="pui-hero-rule"></div>'
        f'<div class="pui-eyebrow">Virginia Tech · Digital Ag Technologies Lab</div>'
        f"<h1>PlantCountGUI</h1>"
        f"<p>Upload a georeferenced orthomosaic, run detection, and review plant counts.</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ---------- health check ----------
backend_ok = True
try:
    hr = requests_sess.get(
        f"{API_BASE}/health",
        timeout=(5, 5),
        allow_redirects=False,
        proxies={"http": None, "https": None},
        headers={"Connection": "close"},
    )
    hr.raise_for_status()
except Exception:
    backend_ok = False

render_hero(backend_ok)

if not backend_ok:
    st.error(
        f"**Cannot reach the detection service at `{API_BASE}`.**\n\n"
        "Start the API (`./run_api_strawberry.sh` or `docker compose up`), then reload this page. "
        "Point the UI somewhere else with the `API_BASE` environment variable."
    )
    st.button("Retry connection", type="primary")
    st.stop()


# ---------- SERVER CALL: cached ----------
@st.cache_data(show_spinner=True, ttl=3600, max_entries=64)
def infer_cached(
    file_bytes: bytes,
    safe_name: str,
    api_base: str,
    apply_dedupe: bool,
    dedupe_radius_m: float,
) -> dict:
    _ = sha1_bytes(file_bytes)
    files = {"file": (safe_name, io.BytesIO(file_bytes), "image/tiff")}
    form = {
        "apply_dedupe": str(bool(apply_dedupe)).lower(),
        "dedupe_radius_m": str(float(dedupe_radius_m)),
    }
    try:
        r = requests_sess.post(
            f"{api_base}/infer",
            files=files,
            data=form,
            timeout=(15, 600),
            allow_redirects=False,
            proxies={"http": None, "https": None},
            headers={"Connection": "close"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e1:
        if SHOW_DEBUG:
            st.info(f"requests upload failed, retrying with toolbelt: {repr(e1)}")

    try:
        bio = io.BytesIO(file_bytes)
        encoder = MultipartEncoder(
            fields={
                "file": (safe_name, bio, "image/tiff"),
                "apply_dedupe": str(bool(apply_dedupe)).lower(),
                "dedupe_radius_m": str(float(dedupe_radius_m)),
            }
        )
        monitor = MultipartEncoderMonitor(encoder, lambda m: None)
        headers = {"Content-Type": monitor.content_type, "Connection": "close"}
        r = requests_sess.post(
            f"{api_base}/infer",
            data=monitor,
            timeout=(15, 600),
            allow_redirects=False,
            proxies={"http": None, "https": None},
            headers=headers,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e2:
        if SHOW_DEBUG:
            st.info(f"toolbelt upload failed, retrying with httpx: {repr(e2)}")

    files = {"file": (safe_name, file_bytes, "image/tiff")}
    data = {
        "apply_dedupe": str(bool(apply_dedupe)).lower(),
        "dedupe_radius_m": str(float(dedupe_radius_m)),
    }
    r = httpx_client.post(
        f"{api_base}/infer", files=files, data=data, headers={"Connection": "close"}
    )
    r.raise_for_status()
    return r.json()

@st.cache_data(show_spinner=True, ttl=3600, max_entries=64)
def infer_csv_cached(
    tiff_bytes: bytes,
    tiff_name: str,
    csv_bytes: bytes,
    csv_name: str,
    api_base: str,
) -> dict:
    files = {
        "file": (tiff_name, io.BytesIO(tiff_bytes), "image/tiff"),
        "detections_csv": (csv_name, io.BytesIO(csv_bytes), "text/csv"),
    }
    try:
        r = requests_sess.post(
            f"{api_base}/infer_csv",
            files=files,
            timeout=(15, 600),
            allow_redirects=False,
            proxies={"http": None, "https": None},
            headers={"Connection": "close"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e1:
        if SHOW_DEBUG:
            st.info(f"requests upload failed, retrying with toolbelt: {repr(e1)}")

    try:
        tiff_bio = io.BytesIO(tiff_bytes)
        csv_bio = io.BytesIO(csv_bytes)
        encoder = MultipartEncoder(
            fields={
                "file": (tiff_name, tiff_bio, "image/tiff"),
                "detections_csv": (csv_name, csv_bio, "text/csv"),
            }
        )
        monitor = MultipartEncoderMonitor(encoder, lambda m: None)
        headers = {"Content-Type": monitor.content_type, "Connection": "close"}
        r = requests_sess.post(
            f"{api_base}/infer_csv",
            data=monitor,
            timeout=(15, 600),
            allow_redirects=False,
            proxies={"http": None, "https": None},
            headers=headers,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e2:
        if SHOW_DEBUG:
            st.info(f"toolbelt upload failed, retrying with httpx: {repr(e2)}")

    files = {
        "file": (tiff_name, tiff_bytes, "image/tiff"),
        "detections_csv": (csv_name, csv_bytes, "text/csv"),
    }
    r = httpx_client.post(f"{api_base}/infer_csv", files=files, headers={"Connection": "close"})
    r.raise_for_status()
    return r.json()


def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    sigma = max(float(sigma), 0.25)
    radius = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= k.sum()
    return k


def _gaussian_blur_2d(arr: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.05 or arr.size == 0:
        return arr.astype(np.float64, copy=True)
    k = _gaussian_kernel_1d(sigma)
    pad = len(k) // 2
    padded = np.pad(arr.astype(np.float64), ((0, 0), (pad, pad)), mode="reflect")
    tmp = np.empty_like(arr, dtype=np.float64)
    for i in range(arr.shape[0]):
        tmp[i] = np.convolve(padded[i], k, mode="valid")
    padded = np.pad(tmp, ((pad, pad), (0, 0)), mode="reflect")
    out = np.empty_like(arr, dtype=np.float64)
    for j in range(arr.shape[1]):
        out[:, j] = np.convolve(padded[:, j], k, mode="valid")
    return out


@st.cache_data(show_spinner=False)
def build_smooth_density(
    utm_x,
    utm_y,
    utm_bounds,
    disp_w: int,
    disp_h: int,
    sigma_m: float,
    max_side: int = 640,
):
    """Point detections → fine 2D density, Gaussian-smoothed for a continuous heatmap."""
    left, bottom, right, top = utm_bounds
    scale = min(1.0, float(max_side) / float(max(disp_w, disp_h, 1)))
    nw = max(48, int(round(disp_w * scale)))
    nh = max(48, int(round(disp_h * scale)))
    dens = np.zeros((nh, nw), dtype=np.float64)
    if len(utm_x) == 0:
        return dens

    utm_x = np.asarray(utm_x, dtype=np.float64)
    utm_y = np.asarray(utm_y, dtype=np.float64)
    x_disp = (utm_x - left) / max(right - left, 1e-9) * disp_w
    y_disp = (top - utm_y) / max(top - bottom, 1e-9) * disp_h
    H, _, _ = np.histogram2d(
        x_disp,
        y_disp,
        bins=[nw, nh],
        range=[[0.0, float(disp_w)], [0.0, float(disp_h)]],
    )
    dens = H.T.astype(np.float64)

    m_per_px = 0.5 * ((right - left) / max(disp_w, 1) + (top - bottom) / max(disp_h, 1))
    sigma_px = max(float(sigma_m) / max(m_per_px, 1e-9) * scale, 0.75)
    dens = _gaussian_blur_2d(dens, sigma_px)
    return dens


@st.cache_data(show_spinner=False)
def decode_thumb_b64_to_np(b64_str: str) -> np.ndarray:
    img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def _to_display_name(s: str) -> str:
    s_norm = str(s).strip().lower()
    if s_norm == "ragweed":
        return "Common Ragweed"
    if s_norm == "palmer":
        return "Palmer Amaranth"
    if s_norm == "grass":
        return "Grass Weeds"
    return re.sub(r"\s+", " ", re.sub(r"_+", " ", str(s))).strip().title()


# ---------- sidebar ----------
SOURCE_YOLO_DEDUPE = "Run YOLO on orthomosaic + dedupe + CSV"
SOURCE_YOLO_RAW = "Run YOLO on orthomosaic (raw, no dedupe)"
SOURCE_CSV = "Plot detections from uploaded CSV"

with st.sidebar:
    st.markdown('<div class="pui-side-title">Workflow</div>', unsafe_allow_html=True)
    for idx, text in enumerate(
        [
            "Upload a georeferenced GeoTIFF",
            "Choose a data source and options",
            "Run detection",
            "Review the map and density hotspots",
            "Export CSV or shapefiles",
        ],
        start=1,
    ):
        st.markdown(
            f'<div class="pui-side-step"><span class="idx">{idx}</span><span>{html.escape(text)}</span></div>',
            unsafe_allow_html=True,
        )
    st.markdown('<div class="pui-side-title" style="margin-top:1.2rem">Service</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="pui-meta"><b>API endpoint</b><br>{html.escape(API_BASE)}<br>'
        f'<span style="color:{OK}">Healthy</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="pui-side-title" style="margin-top:1.2rem">Tips</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="pui-meta">Scroll to zoom and drag to pan either map. '
        "Double-click a map to reset the view, and click a legend entry to hide that class.</div>",
        unsafe_allow_html=True,
    )

# ---------- setup: upload, configure, run ----------
st.session_state.setdefault("infer_data", None)
st.session_state.setdefault("infer_sha", None)
st.session_state.setdefault("infer_name", None)

# Once a run has produced results, tuck the setup away so the numbers lead.
has_results = st.session_state.infer_data is not None
if has_results:
    setup_host = st.expander("Inputs & settings — change the file, switch mode, or re-run", expanded=False)
else:
    setup_host = st.container()

with setup_host:
    section("Input", "Upload the orthomosaic you want to analyze.", step="1")
    with st.container(border=True):
        uploaded = st.file_uploader("GeoTIFF orthomosaic", type=["tif", "tiff"])
        if uploaded is None:
            st.markdown(
                '<div class="pui-hint">Coordinates are read straight from the raster — '
                "no manual alignment needed.</div>",
                unsafe_allow_html=True,
            )

    section("Detection settings", "Choose how detections are produced for this orthomosaic.", step="2")
    with st.container(border=True):
        ctrl_a, ctrl_b = st.columns([1.5, 1])
        with ctrl_a:
            data_source = st.selectbox(
                "Data source",
                (SOURCE_YOLO_DEDUPE, SOURCE_YOLO_RAW, SOURCE_CSV),
                index=0,
                help="YOLO modes run the model on the GeoTIFF. CSV mode skips YOLO and plots existing global_x/global_y.",
            )
        with ctrl_b:
            dedupe_radius_m = 0.5
            if data_source == SOURCE_YOLO_DEDUPE:
                dedupe_radius_m = float(
                    st.number_input(
                        "Dedupe radius (m)",
                        min_value=0.05,
                        max_value=25.0,
                        value=0.5,
                        step=0.05,
                        help="Merge detections within this distance (UTM plane); highest confidence kept.",
                    )
                )
            elif data_source == SOURCE_YOLO_RAW:
                st.markdown(
                    '<div class="pui-hint">Deduplication is off — overlapping tile detections are kept as-is.</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="pui-hint">No model run — counts come straight from your CSV.</div>',
                    unsafe_allow_html=True,
                )

        csv_uploaded = None
        if data_source == SOURCE_CSV:
            csv_uploaded = st.file_uploader(
                "Detections CSV (required for this mode)",
                type=["csv"],
                help="Must include columns global_x and global_y (e.g. deduplicated_detections.csv).",
            )

    section("Run", "Detection can take a few minutes on large mosaics.", step="3")
    run = st.button(
        "Run detection",
        type="primary",
        use_container_width=True,
        disabled=uploaded is None,
        help=None if uploaded is not None else "Upload a GeoTIFF first.",
    )

if uploaded is None:
    st.markdown(
        '<div class="pui-hint">Upload a georeferenced TIFF above to get started.</div>',
        unsafe_allow_html=True,
    )
    how_it_works()
    st.markdown(
        '<div class="pui-foot">PlantCountGUI · Virginia Tech Digital Ag Technologies Lab</div>',
        unsafe_allow_html=True,
    )
    st.stop()

safe_name = sanitize_filename(getattr(uploaded, "name", None))
file_bytes = uploaded.getbuffer().tobytes()
file_sha = sha1_bytes(file_bytes)
csv_bytes = None
csv_name = None
csv_sha = ""
if csv_uploaded is not None:
    csv_name = sanitize_filename(getattr(csv_uploaded, "name", None) or "detections.csv")
    csv_bytes = csv_uploaded.getbuffer().tobytes()
    csv_sha = sha1_bytes(csv_bytes)

yolo_dedupe = data_source == SOURCE_YOLO_DEDUPE
mode_key = f"{data_source}|r{dedupe_radius_m:.3f}" if yolo_dedupe else data_source
run_sig = f"{file_sha}:{csv_sha}:{mode_key}"

if run:
    if data_source == SOURCE_CSV and csv_uploaded is None:
        st.error("Upload a CSV with global_x / global_y, or switch to a YOLO mode.")
        st.stop()
    spinner_msg = (
        "Running YOLO on orthomosaic tiles…"
        if data_source.startswith("Run YOLO")
        else "Loading detections from CSV…"
    )
    with st.spinner(spinner_msg):
        try:
            if data_source.startswith("Run YOLO"):
                data = infer_cached(
                    file_bytes,
                    safe_name,
                    API_BASE,
                    apply_dedupe=yolo_dedupe,
                    dedupe_radius_m=dedupe_radius_m,
                )
            else:
                data = infer_csv_cached(file_bytes, safe_name, csv_bytes, csv_name, API_BASE)
            st.session_state.infer_data = data
            st.session_state.infer_sha = run_sig
            st.session_state.infer_name = safe_name
        except Exception as e:
            st.error(str(e))
            if SHOW_DEBUG:
                st.exception(e)
            st.stop()
    if not has_results:
        # First successful run: rerun so the layout settles into results-first.
        st.rerun()

if st.session_state.infer_data is None:
    st.info("Ready when you are — click **Run detection** to analyze this orthomosaic.")
    how_it_works()
    st.stop()

if st.session_state.infer_sha != run_sig:
    st.warning("The inputs changed since the last run. Click **Run detection** to refresh results.")
    st.stop()

data = st.session_state.infer_data
thumb_np = decode_thumb_b64_to_np(data["thumb_png_b64"])

disp_w, disp_h = data["disp_w"], data["disp_h"]
scale_x, scale_y = data["scale_x"], data["scale_y"]
cell_m = float(int(data["cell_m"]))
left_m, bottom_m, right_m, top_m = data["meta"]["utm_bounds"]

class_names = data.get("class_names", ["ragweed", "grass", "palmer", "soybean"])
totals_by_class = data.get("totals_by_class", [0] * len(class_names))
pix_x_by_class = data.get("pix_x_by_class", [[] for _ in range(len(class_names))])
pix_y_by_class = data.get("pix_y_by_class", [[] for _ in range(len(class_names))])
utm_x_by_class = data.get("utm_x_by_class", [[] for _ in range(len(class_names))])
utm_y_by_class = data.get("utm_y_by_class", [[] for _ in range(len(class_names))])

weed_idx = [i for i, n in enumerate(class_names) if str(n).strip().lower() != "soybean"]
class_names = [class_names[i] for i in weed_idx]
totals_by_class = [totals_by_class[i] for i in weed_idx]
pix_x_by_class = [pix_x_by_class[i] for i in weed_idx]
pix_y_by_class = [pix_y_by_class[i] for i in weed_idx]
utm_x_by_class = [utm_x_by_class[i] for i in weed_idx]
utm_y_by_class = [utm_y_by_class[i] for i in weed_idx]

# Only show classes that contribute to total/density (e.g. strawberry only, not tobacco FPs).
count_names = data.get("count_class_names") or class_names
count_lower = {str(n).strip().lower() for n in count_names}
show_idx = [i for i, n in enumerate(class_names) if str(n).strip().lower() in count_lower]
class_names = [class_names[i] for i in show_idx]
totals_by_class = [totals_by_class[i] for i in show_idx]
pix_x_by_class = [pix_x_by_class[i] for i in show_idx]
pix_y_by_class = [pix_y_by_class[i] for i in show_idx]
utm_x_by_class = [utm_x_by_class[i] for i in show_idx]
utm_y_by_class = [utm_y_by_class[i] for i in show_idx]

NUM_CLASSES = len(class_names)
display_names = [_to_display_name(n) for n in class_names]

# ---------- results ----------
section("Results", f"Detection complete for {safe_name}.")

total_detections = int(data.get("total_weeds", 0))
field_area = float(data.get("field_area_m2", 0.0))
avg_density = float(data.get("avg_density", 0.0))
if len(count_names) == 1:
    total_label = f"{_to_display_name(count_names[0])} detections"
    total_sub = "Counted across the full mosaic"
else:
    total_label = "Total detections"
    total_sub = f"Across {NUM_CLASSES} class{'es' if NUM_CLASSES != 1 else ''}"

dominant = display_names[int(np.argmax(totals_by_class))] if NUM_CLASSES else "N/A"
dominant_share = (
    f"{(max(totals_by_class) / total_detections * 100):.0f}% of detections"
    if NUM_CLASSES and total_detections > 0
    else "—"
)

kpi_row(
    [
        {"label": total_label, "value": f"{total_detections:,}", "sub": total_sub, "accent": ACCENT},
        {
            "label": "Field area",
            "value": f"{field_area:,.0f}",
            "unit": "m²",
            "sub": f"{field_area / 10000.0:,.2f} ha surveyed",
            "accent": "#3FD0C9",
        },
        {
            "label": "Average density",
            "value": f"{avg_density:.4f}",
            "unit": "/m²",
            "sub": f"≈ {avg_density * 10000.0:,.0f} per hectare",
            "accent": "#8FD14F",
        },
        {"label": "Dominant class", "value": dominant, "sub": dominant_share, "accent": "#9B8CFF"},
    ]
)

class_chips(display_names, totals_by_class)

with st.expander("Run metadata", expanded=False):
    meta = data.get("meta") or {}
    meta_a, meta_b = st.columns(2)
    with meta_a:
        st.markdown(f"**Source file** · {safe_name}")
        st.markdown(f"**Mode** · {data_source}")
        st.markdown(f"**Model classes** · {', '.join(display_names) if display_names else 'N/A'}")
        if count_names != class_names:
            st.markdown(
                f"**Counted in total/density** · {', '.join(_to_display_name(n) for n in count_names)}"
            )
    with meta_b:
        st.markdown(f"**Base grid cell** · {int(cell_m)} m")
        if meta.get("utm_epsg"):
            st.markdown(f"**Projection** · EPSG:{int(meta['utm_epsg'])} (UTM)")
        if meta.get("width") and meta.get("height"):
            st.markdown(f"**Raster size** · {int(meta['width']):,} × {int(meta['height']):,} px")
        if meta.get("px_w"):
            st.markdown(f"**Ground sample distance** · {float(meta['px_w']) * 100:.2f} cm/px")

# A breakdown of a single class only restates the headline count, so it is
# shown when there is more than one class to compare.
if NUM_CLASSES > 1:
    section("Class breakdown", "How detections are distributed across the counted classes.")
    order = sorted(range(NUM_CLASSES), key=lambda i: int(totals_by_class[i]))
    chart_h = max(220, 62 * NUM_CLASSES + 90)

    chart_col, table_col = st.columns([1.35, 1])
    with chart_col:
        with st.container(border=True):
            fig_bar = go.Figure(
                go.Bar(
                    x=[int(totals_by_class[i]) for i in order],
                    y=[display_names[i] for i in order],
                    orientation="h",
                    marker=dict(
                        color=[PALETTE[i % len(PALETTE)] for i in order],
                        line=dict(width=0),
                        cornerradius=8,
                    ),
                    text=[f"{int(totals_by_class[i]):,}" for i in order],
                    textposition="outside",
                    cliponaxis=False,
                    hovertemplate="<b>%{y}</b><br>%{x:,} detections<extra></extra>",
                )
            )
            fig_bar.update_layout(
                showlegend=False,
                height=chart_h,
                margin=dict(l=10, r=46, t=18, b=10),
                bargap=0.45,
            )
            fig_bar.update_xaxes(
                title_text="Detections",
                rangemode="tozero",
                range=[0, max(1, max(int(v) for v in totals_by_class)) * 1.18],
            )
            fig_bar.update_yaxes(title_text=None, ticksuffix="  ")
            fig_bar = bump_bar_label_fonts(fig_bar)
            fig_bar = apply_plotly_fonts(fig_bar)
            st.plotly_chart(fig_bar, use_container_width=True, config=PLOTLY_CONFIG_STATIC)
    with table_col:
        with st.container(border=True):
            share = [
                (int(v) / total_detections * 100.0) if total_detections else 0.0
                for v in totals_by_class
            ]
            st.dataframe(
                {
                    "Class": display_names,
                    "Detections": [int(v) for v in totals_by_class],
                    "% of total": share,
                },
                hide_index=True,
                use_container_width=True,
                height=min(chart_h - 44, 36 * NUM_CLASSES + 40),
                column_config={
                    "Detections": st.column_config.NumberColumn("Detections", format="%d"),
                    "% of total": st.column_config.ProgressColumn(
                        "% of total",
                        help="This class as a percentage of all counted detections.",
                        format="%.1f%%",
                        min_value=0.0,
                        max_value=100.0,
                    ),
                },
            )

section("Spatial review", "Inspect where detections fall, where they cluster, and export the layers.")
viz_tab, heat_tab, export_tab = st.tabs(["Detections map", "Density heatmap", "Export"])

with viz_tab:
    ctl_col, marker_col, count_col = st.columns([2.4, 1.1, 1])
    with ctl_col:
        chosen = st.multiselect(
            "Visible classes",
            options=list(range(NUM_CLASSES)),
            default=list(range(NUM_CLASSES)),
            format_func=lambda i: display_names[i],
            help="Hide classes to declutter the map. The count on the right follows this filter.",
        )
    with marker_col:
        marker_size = st.slider("Marker size", 4, 16, 7)
    visible_total = int(sum(int(totals_by_class[ci]) for ci in chosen))
    with count_col:
        st.metric(
            "Plants shown",
            f"{visible_total:,}",
            delta=None if visible_total == total_detections else f"{visible_total - total_detections:,} vs total",
            delta_color="off",
        )

    if not chosen:
        st.info("Select at least one class to plot detections.")

    fig_points = px.imshow(thumb_np)
    fig_points.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
    fig_points.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
    fig_points.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_points.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        dragmode="pan",
        height=760,
        legend=dict(orientation="h", yanchor="bottom", y=1.005, xanchor="left", x=0),
    )

    for ci in chosen:
        if not pix_x_by_class[ci]:
            continue
        sx = np.asarray(pix_x_by_class[ci], dtype=np.float64) * scale_x
        sy = np.asarray(pix_y_by_class[ci], dtype=np.float64) * scale_y
        fig_points.add_trace(
            go.Scattergl(
                x=sx,
                y=sy,
                mode="markers",
                name=f"{display_names[ci]} ({len(sx):,})",
                marker=dict(
                    size=marker_size,
                    color=PALETTE[ci % len(PALETTE)],
                    opacity=0.95,
                    line=dict(width=1.2, color="rgba(8, 11, 10, 0.85)"),
                ),
                hovertemplate=f"{display_names[ci]}<extra></extra>",
                showlegend=True,
            )
        )

    fig_points = apply_plotly_fonts(fig_points)
    st.plotly_chart(fig_points, use_container_width=True, config=PLOTLY_CONFIG_MAP)
    st.markdown(
        '<div class="pui-hint">Scroll to zoom · drag to pan · double-click to reset.</div>',
        unsafe_allow_html=True,
    )

with heat_tab:
    scope_col, smooth_col, opacity_col = st.columns([1.3, 1, 1])
    with scope_col:
        heat_classes_mode = st.radio(
            "Scope",
            ["All classes", "Single class"],
            horizontal=True,
        )
        if heat_classes_mode == "Single class" and NUM_CLASSES:
            single_class = st.selectbox(
                "Class",
                options=list(range(NUM_CLASSES)),
                format_func=lambda i: display_names[i],
            )
            target_classes = [single_class]
        else:
            target_classes = list(range(NUM_CLASSES))
    with smooth_col:
        sigma_m = st.slider(
            "Smoothness (m)",
            min_value=0.5,
            max_value=float(max(8.0, 8 * cell_m)),
            value=0.5,
            step=0.5,
            help="Larger = softer hotspots. Smaller = tighter around detections.",
        )
    with opacity_col:
        heat_alpha = st.slider(
            "Overlay opacity", 0.0, 1.0, 0.80, 0.05, help="Fade the heatmap to see the imagery beneath."
        )

    all_utm_x = []
    all_utm_y = []
    for ci in target_classes:
        all_utm_x.extend(utm_x_by_class[ci])
        all_utm_y.extend(utm_y_by_class[ci])

    dens = build_smooth_density(
        all_utm_x,
        all_utm_y,
        (left_m, bottom_m, right_m, top_m),
        int(disp_w),
        int(disp_h),
        float(sigma_m),
    )
    dens_plot = dens.astype(np.float64, copy=True)
    peak = float(np.nanmax(dens_plot)) if dens_plot.size else 0.0
    if peak > 0:
        dens_plot[dens_plot < peak * 0.04] = np.nan
    zmax_val = peak if peak > 0 else 1.0

    fig_heat = px.imshow(thumb_np)
    fig_heat.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
    fig_heat.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
    fig_heat.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_heat.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode="pan", height=760)

    nh, nw = dens_plot.shape
    fig_heat.add_trace(
        go.Heatmap(
            z=dens_plot,
            x0=0.0,
            dx=float(disp_w) / max(nw, 1),
            y0=0.0,
            dy=float(disp_h) / max(nh, 1),
            colorscale=[
                [0.0, "rgba(48, 18, 59, 0)"],
                [0.12, "rgba(70, 52, 160, 0.45)"],
                [0.28, "rgba(33, 144, 140, 0.6)"],
                [0.45, "rgba(94, 201, 98, 0.72)"],
                [0.62, "rgba(253, 231, 37, 0.82)"],
                [0.8, "rgba(253, 141, 60, 0.9)"],
                [1.0, "rgba(215, 25, 28, 0.95)"],
            ],
            zmin=0,
            zmax=zmax_val,
            opacity=float(heat_alpha),
            zsmooth="best",
            hoverongaps=False,
            colorbar=dict(
                title=dict(text="Density", side="right"),
                thickness=14,
                len=0.72,
                outlinewidth=0,
                tickfont=dict(color=PLOT_INK),
            ),
            hovertemplate="density: %{z:.3f}<extra></extra>",
            name="Density",
        )
    )
    fig_heat = apply_plotly_fonts(fig_heat)
    st.plotly_chart(fig_heat, use_container_width=True, config=PLOTLY_CONFIG_MAP)
    st.markdown(
        f'<div class="pui-hint">Warm colors mark the densest clusters of '
        f'{html.escape(", ".join(display_names[ci] for ci in target_classes)) or "detections"}.</div>',
        unsafe_allow_html=True,
    )

with export_tab:
    dedup_csv_b64 = (data.get("meta") or {}).get("dedup_csv_b64")
    shp_zip_all = data.get("shp_zip_b64_all")
    shp_zip_by_class = data.get("shp_zip_b64_by_class", [])

    exp_a, exp_b = st.columns(2)
    with exp_a:
        with st.container(border=True):
            st.markdown("**Detections table (CSV)**")
            st.markdown(
                '<div class="pui-hint">One row per detection with UTM coordinates — ready for '
                "spreadsheets or re-plotting in CSV mode.</div>",
                unsafe_allow_html=True,
            )
            if dedup_csv_b64:
                st.download_button(
                    label="Download deduplicated_detections.csv",
                    data=base64.b64decode(dedup_csv_b64),
                    file_name="deduplicated_detections.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    '<div class="pui-hint">Not available for this run.</div>', unsafe_allow_html=True
                )
    with exp_b:
        with st.container(border=True):
            st.markdown("**All detections (shapefile)**")
            st.markdown(
                '<div class="pui-hint">Zipped point shapefile of every counted detection, '
                "for QGIS / ArcGIS.</div>",
                unsafe_allow_html=True,
            )
            if shp_zip_all:
                st.download_button(
                    label="Download detections_all.zip",
                    data=base64.b64decode(shp_zip_all),
                    file_name="detections_all.zip",
                    mime="application/zip",
                    use_container_width=True,
                )
            else:
                st.markdown(
                    '<div class="pui-hint">Not available for this run.</div>', unsafe_allow_html=True
                )

    if NUM_CLASSES:
        st.markdown(
            '<div class="pui-hint" style="margin-top:0.9rem"><b>Per-class shapefiles</b></div>',
            unsafe_allow_html=True,
        )
        cols = st.columns(max(1, min(NUM_CLASSES, 4)))
        for i in range(NUM_CLASSES):
            with cols[i % len(cols)]:
                has_blob = i < len(shp_zip_by_class) and shp_zip_by_class[i]
                st.download_button(
                    label=f"{display_names[i]} ({int(totals_by_class[i]):,})",
                    data=base64.b64decode(shp_zip_by_class[i]) if has_blob else b"",
                    file_name=f"detections_{class_names[i]}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    disabled=not has_blob,
                )

st.markdown(
    '<div class="pui-foot">PlantCountGUI · Virginia Tech Digital Ag Technologies Lab</div>',
    unsafe_allow_html=True,
)
