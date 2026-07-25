import os
import re
import urllib.parse
import io
import base64
import hashlib

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

import requests
import httpx
from requests_toolbelt.multipart.encoder import MultipartEncoder, MultipartEncoderMonitor
import streamlit as st

st.set_page_config(
    page_title="Plant Counting",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_FONT = int(os.getenv("BASE_FONT_PT", "15"))
SHOW_DEBUG = bool(int(os.getenv("UI_DEBUG", "0")))
BOLD_FAMILY = os.getenv("BOLD_FONT_FAMILY", "IBM Plex Sans, Segoe UI, sans-serif")
BAR_TEXT_MULT = float(os.getenv("BAR_TEXT_MULT", "1.1"))


def bump_bar_label_fonts(fig, base=BASE_FONT):
    fig.update_traces(
        selector=dict(type="bar"),
        textfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
        insidetextfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
        outsidetextfont=dict(size=int(base * BAR_TEXT_MULT), family=BOLD_FAMILY),
    )
    fig.update_layout(uniformtext_minsize=int(base * BAR_TEXT_MULT), uniformtext_mode="show")
    return fig


def bump_streamlit_fonts(base=BASE_FONT):
    # Prefer components.html so Streamlit never renders raw CSS as page text.
    import streamlit.components.v1 as components

    css = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,500;9..144,700&family=IBM+Plex+Sans:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {{
  --ink: #e9f2ec;
  --muted: #9db4a8;
  --surface: #0b1310;
  --panel: rgba(255, 255, 255, 0.035);
  --panel-strong: #122019;
  --line: rgba(233, 242, 236, 0.10);
  --accent: #34d399;
  --accent-deep: #1f7a4d;
  --accent-2: #e0813f;
  --map-bg: #0a110e;
}}

.stApp {{
  font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
  font-size: {base}px;
  color: var(--ink);
  background:
    radial-gradient(1100px 520px at 10% -12%, rgba(52, 211, 153, 0.10), transparent 55%),
    radial-gradient(900px 460px at 104% -6%, rgba(224, 129, 63, 0.09), transparent 50%),
    linear-gradient(180deg, #0d1712 0%, #0a120e 45%, #0b130f 100%);
}}

header[data-testid="stHeader"] {{ background: transparent; }}
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] {{
  visibility: hidden;
  height: 0;
}}

::-webkit-scrollbar {{ width: 10px; height: 10px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{
  background: rgba(233, 242, 236, 0.14);
  border-radius: 999px;
  border: 2px solid transparent;
  background-clip: padding-box;
}}
::-webkit-scrollbar-thumb:hover {{ background: rgba(233, 242, 236, 0.26); background-clip: padding-box; }}

.stApp a {{ color: var(--accent); }}
.stApp hr {{ border-color: var(--line); }}

[data-testid="stSidebar"] {{
  background: linear-gradient(180deg, #0f2b1f 0%, #0a1d16 55%, #0c2018 100%);
  border-right: 1px solid rgba(255,255,255,0.06);
}}
[data-testid="stSidebar"] * {{ color: #e8f0ea !important; }}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] label {{ color: rgba(232,240,234,0.78) !important; }}
[data-testid="stSidebar"] hr {{ border-color: rgba(255,255,255,0.12); }}

h1, h2, h3, .brand-title {{
  font-family: "Fraunces", Georgia, serif !important;
  letter-spacing: -0.02em;
  color: var(--ink) !important;
}}
h1 {{ font-size: {int(base*2.05)}px !important; }}
h2 {{ font-size: {int(base*1.45)}px !important; }}
h3 {{ font-size: {int(base*1.18)}px !important; }}

.stButton>button {{
  border-radius: 10px;
  font-weight: 600;
  border: 1px solid var(--line);
  background: var(--panel);
  color: var(--ink);
  transition: transform .15s ease, box-shadow .15s ease, border-color .15s ease;
}}
.stButton>button[kind="primary"], .stButton>button[data-testid="baseButton-primary"] {{
  background: linear-gradient(180deg, #2fbf77 0%, #1f7a4d 100%);
  border-color: rgba(52, 211, 153, 0.45);
  color: #06130c;
  box-shadow: 0 10px 26px rgba(47, 191, 119, 0.28);
}}
.stButton>button:hover {{ transform: translateY(-1px); border-color: rgba(52, 211, 153, 0.5); }}
.stDownloadButton>button {{
  border-radius: 10px;
  font-weight: 600;
  background: var(--panel);
  border: 1px solid var(--line);
  color: var(--ink);
  transition: transform .15s ease, border-color .15s ease, box-shadow .15s ease;
}}
.stDownloadButton>button:hover {{
  transform: translateY(-1px);
  border-color: rgba(52, 211, 153, 0.5);
  box-shadow: 0 8px 22px rgba(0, 0, 0, 0.35);
}}

.stFileUploader {{ border-radius: 12px; }}
[data-testid="stFileUploaderDropzone"] {{
  background: rgba(255, 255, 255, 0.03);
  border: 1.5px dashed rgba(233, 242, 236, 0.20);
  border-radius: 14px;
  transition: border-color .2s ease, background .2s ease;
}}
[data-testid="stFileUploaderDropzone"]:hover {{
  border-color: rgba(52, 211, 153, 0.55);
  background: rgba(52, 211, 153, 0.05);
}}

div[data-testid="stMetric"] {{
  position: relative;
  overflow: hidden;
  background: linear-gradient(180deg, rgba(255,255,255,0.05) 0%, rgba(255,255,255,0.02) 100%);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 0.85rem 1rem 0.7rem;
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.25);
}}
div[data-testid="stMetric"]::before {{
  content: "";
  position: absolute;
  top: 0; left: 0; right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--accent), rgba(52, 211, 153, 0) 75%);
  opacity: 0.85;
}}
div[data-testid="stMetric"] label {{ color: var(--muted) !important; font-weight: 500; }}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
  font-family: "Fraunces", Georgia, serif;
  font-weight: 700;
  color: var(--ink);
}}

.stTabs [data-baseweb="tab-list"] {{
  gap: 0.35rem;
  background: rgba(255,255,255,0.04);
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 0.3rem;
}}
.stTabs [data-baseweb="tab"] {{
  border-radius: 9px;
  padding: 0.55rem 1rem;
  font-weight: 600;
  color: var(--muted);
}}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(180deg, #2fbf77 0%, #1f7a4d 100%) !important;
  color: #06130c !important;
}}

[data-testid="stExpander"] {{
  border: 1px solid var(--line);
  border-radius: 12px;
  background: rgba(255,255,255,0.03);
  overflow: hidden;
}}
[data-testid="stExpander"] details {{ border: none; }}

[data-testid="stDataFrame"] {{
  border: 1px solid var(--line);
  border-radius: 12px;
  overflow: hidden;
}}

div[data-testid="stAlert"] {{ border-radius: 12px; }}

.hero {{
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(620px 240px at 88% -10%, rgba(224, 129, 63, 0.30), transparent 62%),
    radial-gradient(720px 300px at 8% 115%, rgba(52, 211, 153, 0.22), transparent 60%),
    linear-gradient(120deg, #10271d 0%, #0e2018 55%, #14231c 100%);
  border: 1px solid rgba(233, 242, 236, 0.10);
  border-radius: 18px;
  padding: 1.35rem 1.5rem 1.25rem;
  color: var(--ink);
  margin-bottom: 1.1rem;
  box-shadow: 0 24px 50px rgba(0, 0, 0, 0.35);
}}
.hero::before {{
  content: "";
  position: absolute;
  top: -45%; right: -8%;
  width: 420px; height: 420px;
  background: radial-gradient(circle, rgba(52, 211, 153, 0.18), transparent 62%);
  animation: hero-drift 16s ease-in-out infinite alternate;
  pointer-events: none;
}}
@keyframes hero-drift {{
  from {{ transform: translate3d(0, 0, 0); }}
  to {{ transform: translate3d(-70px, 36px, 0); }}
}}
@media (prefers-reduced-motion: reduce) {{
  .hero::before {{ animation: none; }}
  .status-pill::before {{ animation: none; }}
}}
.hero .eyebrow {{
  display: inline-block;
  font-size: 0.72rem;
  font-weight: 600;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8fe3bd;
  margin-bottom: 0.35rem;
}}
.hero h2 {{
  margin: 0 0 .4rem;
  color: #f4faf6 !important;
  font-size: {int(base*1.85)}px !important;
  position: relative;
  z-index: 1;
}}
.hero p {{
  margin: 0;
  color: rgba(233, 242, 236, 0.82);
  max-width: 52rem;
  position: relative;
  z-index: 1;
}}

.section-head {{
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  gap: 1rem;
  margin: 0.35rem 0 0.65rem;
}}
.section-head h3 {{
  margin: 0;
  font-size: {int(base*1.15)}px !important;
}}
.section-head span {{
  color: var(--muted);
  font-size: 0.86rem;
}}

.viewer {{
  background: var(--map-bg);
  border: 1px solid rgba(233, 242, 236, 0.08);
  border-radius: 16px;
  padding: 0.55rem;
  box-shadow: 0 20px 44px rgba(0, 0, 0, 0.45);
}}

.status-pill {{
  display: inline-flex;
  align-items: center;
  gap: 0.4rem;
  background: rgba(52, 211, 153, 0.10);
  color: #7fe0b4;
  border: 1px solid rgba(52, 211, 153, 0.30);
  border-radius: 999px;
  padding: 0.28rem 0.75rem;
  font-size: 0.82rem;
  font-weight: 600;
  margin-bottom: 0.85rem;
}}
.status-pill::before {{
  content: "";
  width: 0.45rem;
  height: 0.45rem;
  border-radius: 50%;
  background: var(--accent);
  box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.18);
  animation: pill-pulse 2.4s ease-in-out infinite;
}}
@keyframes pill-pulse {{
  0%, 100% {{ box-shadow: 0 0 0 3px rgba(52, 211, 153, 0.18); }}
  50% {{ box-shadow: 0 0 0 6px rgba(52, 211, 153, 0.06); }}
}}

.chip {{
  display: inline-block;
  border-radius: 8px;
  padding: 0.2rem 0.6rem;
  font-size: {int(base*0.82)}px;
  font-weight: 600;
  border: 1px solid var(--line);
  margin: 0 0.35rem 0.35rem 0;
  color: var(--ink);
  background: rgba(255, 255, 255, 0.03);
}}

.export-note {{
  color: var(--muted);
  margin-bottom: 0.75rem;
}}

div[data-testid="column"] .stSlider {{
  padding-top: 0.15rem;
}}
</style>
<script>
(() => {{
  const src = document.currentScript.previousElementSibling;
  if (!src || !window.parent || !window.parent.document) return;
  const doc = window.parent.document;
  doc.querySelectorAll('style[data-plant-counting-ui]').forEach((n) => n.remove());
  const style = doc.createElement('style');
  style.setAttribute('data-plant-counting-ui', '1');
  style.textContent = src.textContent || '';
  doc.head.appendChild(style);
}})();
</script>
"""
    components.html(css, height=0)


PLOT_INK = "#e9f2ec"
PLOT_GRID = "rgba(233, 242, 236, 0.08)"


def apply_plotly_fonts(fig, base=BASE_FONT):
    fig.update_layout(
        font=dict(size=base, color=PLOT_INK, family=BOLD_FAMILY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            font=dict(size=int(base * 0.9), color=PLOT_INK),
            bgcolor="rgba(10, 17, 14, 0.55)",
            bordercolor="rgba(233, 242, 236, 0.12)",
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
        title_font=dict(size=int(base * 1.0), color=PLOT_INK, family=BOLD_FAMILY),
        tickfont=dict(size=int(base * 0.9), color=PLOT_INK, family=BOLD_FAMILY),
        gridcolor=PLOT_GRID,
        zerolinecolor=PLOT_GRID,
    )
    fig.update_yaxes(
        title_font=dict(size=int(base * 1.0), color=PLOT_INK, family=BOLD_FAMILY),
        tickfont=dict(size=int(base * 0.9), color=PLOT_INK, family=BOLD_FAMILY),
        gridcolor=PLOT_GRID,
        zerolinecolor=PLOT_GRID,
    )
    return fig


bump_streamlit_fonts()


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


# ---------- health check ----------
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
    st.error("Cannot reach the backend service. Make sure the API is running at API_BASE.")
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


# ---------- UI ----------
st.markdown(
    """
    <div class='hero'>
      <div class='eyebrow'>Field intelligence</div>
      <h2>Plant Counting</h2>
      <p>Upload a georeferenced orthomosaic, run detection, and inspect counts and density hotspots in one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("### Workspace")
    st.caption("Upload → detect → review → export")
    st.markdown("---")
    st.caption(f"API · {API_BASE}")

uploaded = st.file_uploader("GeoTIFF orthomosaic", type=["tif", "tiff"])

ctrl_a, ctrl_b = st.columns([1.4, 1])
with ctrl_a:
    data_source = st.selectbox(
        "Data source",
        (
            "Run YOLO on orthomosaic + dedupe + CSV",
            "Run YOLO on orthomosaic (raw, no dedupe)",
            "Plot detections from uploaded CSV",
        ),
        index=0,
        help="YOLO modes run the model on the GeoTIFF. CSV mode skips YOLO and plots existing global_x/global_y.",
    )
with ctrl_b:
    dedupe_radius_m = 0.5
    if data_source == "Run YOLO on orthomosaic + dedupe + CSV":
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
    else:
        st.caption("Dedupe off for this mode")

csv_uploaded = None
if data_source == "Plot detections from uploaded CSV":
    csv_uploaded = st.file_uploader(
        "Detections CSV (required for this mode)",
        type=["csv"],
        help="Must include columns global_x and global_y (e.g. deduplicated_detections.csv).",
    )

run = st.button("Run detection", type="primary", use_container_width=True)

if uploaded is None:
    st.info("Upload a georeferenced TIFF to begin.")
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

yolo_dedupe = data_source == "Run YOLO on orthomosaic + dedupe + CSV"
mode_key = f"{data_source}|r{dedupe_radius_m:.3f}" if yolo_dedupe else data_source
run_sig = f"{file_sha}:{csv_sha}:{mode_key}"

st.session_state.setdefault("infer_data", None)
st.session_state.setdefault("infer_sha", None)
st.session_state.setdefault("infer_name", None)

if run:
    if data_source == "Plot detections from uploaded CSV" and csv_uploaded is None:
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

if st.session_state.infer_data is None:
    st.info("Click **Run detection** after uploading the TIFF.")
    st.stop()

if st.session_state.infer_sha != run_sig:
    st.warning("You changed the TIFF or CSV. Click **Run detection** to refresh results.")
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

palette = ["#e8834f", "#3ecf8e", "#6b9bff", "#b18cff", "#f2b25c", "#3fd6c5", "#f07bb1"]

st.markdown("<div class='status-pill'>Detection complete</div>", unsafe_allow_html=True)

m1, m2, m3, m4 = st.columns(4)
if len(count_names) == 1:
    total_label = f"{_to_display_name(count_names[0])} detections"
else:
    total_label = "Total detections"
with m1:
    st.metric(total_label, int(data.get("total_weeds", 0)))
with m2:
    st.metric("Field area (m²)", f"{float(data.get('field_area_m2', 0.0)):,.0f}")
with m3:
    st.metric("Average density", f"{float(data.get('avg_density', 0.0)):.4f} /m²")
with m4:
    dominant = display_names[int(np.argmax(totals_by_class))] if NUM_CLASSES else "N/A"
    st.metric("Dominant class", dominant)

chips = []
for i, name in enumerate(display_names):
    chips.append(f"<span class='chip' style='background:{palette[i % len(palette)]}18;border-color:{palette[i % len(palette)]}55'>{name}</span>")
st.markdown("".join(chips) if chips else "", unsafe_allow_html=True)

with st.expander("Run metadata", expanded=False):
    st.write(f"**Model classes:** {', '.join(display_names) if display_names else 'N/A'}")
    if count_names != display_names:
        st.write(f"**Counted in total/density:** {', '.join(_to_display_name(n) for n in count_names)}")
    st.write(f"**Base grid cell:** {int(cell_m)} m")
    st.write(f"**Filename:** {safe_name}")

if NUM_CLASSES:
    st.markdown(
        "<div class='section-head'><h3>Class breakdown</h3><span>Counts by detected class</span></div>",
        unsafe_allow_html=True,
    )
    chart_col, table_col = st.columns([1.25, 1])
    with chart_col:
        fig_bar = px.bar(
            x=display_names,
            y=[int(v) for v in totals_by_class],
            color=display_names,
            color_discrete_sequence=palette,
            labels={"x": "Class", "y": "Detections"},
            text=[int(v) for v in totals_by_class],
        )
        fig_bar.update_layout(showlegend=False, height=320, title=None, margin=dict(l=10, r=10, t=10, b=10))
        fig_bar = bump_bar_label_fonts(fig_bar)
        fig_bar = apply_plotly_fonts(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
    with table_col:
        st.dataframe(
            {"Class": display_names, "Count": [int(v) for v in totals_by_class]},
            hide_index=True,
            use_container_width=True,
            height=320,
        )

st.markdown(
    "<div class='section-head'><h3>Spatial review</h3><span>Map detections, density, and exports</span></div>",
    unsafe_allow_html=True,
)
viz_tab, heat_tab, export_tab = st.tabs(["Detections map", "Density heatmap", "Export"])

with viz_tab:
    roi_left_pct, roi_top_pct, roi_right_pct, roi_bottom_pct = 5, 10, 95, 90
    x_lo = disp_w * min(roi_left_pct, roi_right_pct) / 100.0
    x_hi = disp_w * max(roi_left_pct, roi_right_pct) / 100.0
    y_lo = disp_h * min(roi_top_pct, roi_bottom_pct) / 100.0
    y_hi = disp_h * max(roi_top_pct, roi_bottom_pct) / 100.0

    class_col, marker_col, count_col = st.columns([2.4, 1, 1])
    with class_col:
        chosen = st.multiselect(
            "Visible classes",
            options=list(range(NUM_CLASSES)),
            default=list(range(NUM_CLASSES)),
            format_func=lambda i: display_names[i],
        )
    with marker_col:
        marker_size = st.slider("Marker size", 4, 16, 7)

    frame_counts = []
    for ci in chosen:
        if not pix_x_by_class[ci]:
            frame_counts.append(0)
            continue
        sx = np.asarray(pix_x_by_class[ci], dtype=np.float64) * scale_x
        sy = np.asarray(pix_y_by_class[ci], dtype=np.float64) * scale_y
        in_frame = (sx >= x_lo) & (sx <= x_hi) & (sy >= y_lo) & (sy <= y_hi)
        frame_counts.append(int(in_frame.sum()))

    frame_total = int(sum(frame_counts))
    with count_col:
        st.metric("Plant count", frame_total)

    fig_points = px.imshow(thumb_np)
    fig_points.update_xaxes(range=[0, disp_w], showgrid=False, visible=False)
    fig_points.update_yaxes(range=[disp_h, 0], showgrid=False, visible=False)
    fig_points.update_yaxes(scaleanchor="x", scaleratio=1)
    fig_points.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False, height=780)

    for ci in chosen:
        if not pix_x_by_class[ci]:
            continue
        sx = np.asarray(pix_x_by_class[ci], dtype=np.float64) * scale_x
        sy = np.asarray(pix_y_by_class[ci], dtype=np.float64) * scale_y
        in_frame = (sx >= x_lo) & (sx <= x_hi) & (sy >= y_lo) & (sy <= y_hi)
        if not np.any(in_frame):
            continue
        fig_points.add_trace(
            go.Scattergl(
                x=sx[in_frame],
                y=sy[in_frame],
                mode="markers",
                name=display_names[ci],
                marker=dict(size=marker_size, color=palette[ci % len(palette)], opacity=0.95, line=dict(width=1, color="white")),
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig_points = apply_plotly_fonts(fig_points)
    st.markdown("<div class='viewer'>", unsafe_allow_html=True)
    st.plotly_chart(fig_points, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with heat_tab:
    scope_col, smooth_col, opacity_col = st.columns([1.1, 1, 1])
    with scope_col:
        heat_classes_mode = st.radio(
            "Scope",
            ["All classes", "Single class"],
            horizontal=True,
            label_visibility="visible",
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
        heat_alpha = st.slider("Opacity", 0.0, 1.0, 0.80, 0.05)

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
    fig_heat.update_layout(margin=dict(l=0, r=0, t=0, b=0), dragmode=False, height=780)

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
            colorbar=dict(title="Density", thickness=14, len=0.72),
            hovertemplate="density: %{z:.3f}<extra></extra>",
            name="Density",
        )
    )
    fig_heat = apply_plotly_fonts(fig_heat)
    st.markdown("<div class='viewer'>", unsafe_allow_html=True)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with export_tab:
    st.markdown("<p class='export-note'>Download detection layers for GIS workflows.</p>", unsafe_allow_html=True)
    dedup_csv_b64 = (data.get("meta") or {}).get("dedup_csv_b64")
    if dedup_csv_b64:
        st.markdown("**Deduplicated detections (CSV)**")
        st.download_button(
            label="Download deduplicated_detections.csv",
            data=base64.b64decode(dedup_csv_b64),
            file_name="deduplicated_detections.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.markdown("---")
    shp_zip_all = data.get("shp_zip_b64_all")
    shp_zip_by_class = data.get("shp_zip_b64_by_class", [])

    if shp_zip_all:
        st.download_button(
            label="Download all detections",
            data=base64.b64decode(shp_zip_all),
            file_name="detections_all.zip",
            mime="application/zip",
            use_container_width=True,
        )

    if NUM_CLASSES:
        st.caption("Per-class shapefiles")
        cols = st.columns(max(1, min(NUM_CLASSES, 4)))
        for i in range(NUM_CLASSES):
            with cols[i % len(cols)]:
                has_blob = i < len(shp_zip_by_class) and shp_zip_by_class[i]
                st.download_button(
                    label=f"{display_names[i]}",
                    data=base64.b64decode(shp_zip_by_class[i]) if has_blob else b"",
                    file_name=f"detections_{class_names[i]}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    disabled=not has_blob,
                )
