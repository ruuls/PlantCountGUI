import os
import io
import base64
import shutil
import hashlib
from tempfile import NamedTemporaryFile, mkdtemp
from pathlib import Path
from typing import List, Tuple

import csv as csv_std
from collections import defaultdict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ultralytics import settings, YOLO

import torch
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.vrt import WarpedVRT
from pyproj import CRS, Transformer
from PIL import Image

from functools import lru_cache
from joblib import Memory

# -------------------------
# ENV / Paths
# -------------------------
os.environ.setdefault("YOLO_CONFIG_DIR", "/tmp/ultralytics")
os.environ.setdefault("ULTRALYTICS_RUNS_DIR", "/tmp/runs")

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent

# Default weights; override anytime, e.g. export MODEL_PATH=/path/to/best.pt
_DEFAULT_WEIGHTS = Path.home() / "Downloads" / "best.pt"
_MODEL_PATH_ENV = os.getenv(
    "MODEL_PATH",
    str(_DEFAULT_WEIGHTS) if _DEFAULT_WEIGHTS.is_file() else str(PROJECT_ROOT / "models" / "bigmodel.pt"),
)
MODEL_PATH = Path(os.path.expanduser(_MODEL_PATH_ENV)).resolve()
TILE_SIZE = int(os.getenv("TILE_SIZE", "640"))
BATCH_TILES = int(os.getenv("BATCH_TILES", "8"))
MAX_DISPLAY = int(os.getenv("MAX_DISPLAY", "2048"))
MAX_CELLS = int(os.getenv("MAX_CELLS", "60000000"))

YOLO_CONF = float(os.getenv("YOLO_CONF", "0.25"))
YOLO_IOU  = float(os.getenv("YOLO_IOU",  "0.45"))

# Startup fallback labels; real labels/classes are synced from the loaded checkpoint in load_model().
CLASS_NAMES = os.getenv("CLASS_NAMES", "plant").split(",")
CLASS_NAMES = [c.strip() for c in CLASS_NAMES if c.strip()]
NUM_CLASSES = len(CLASS_NAMES) or 1

MAX_OVERLAY_POINTS_PER_CLASS = int(os.getenv("MAX_OVERLAY_POINTS_PER_CLASS", "120000"))
DEFAULT_DEDUPE_RADIUS_M = float(os.getenv("DEDUPE_RADIUS_M", "0.5"))
# Comma-separated classes to include in total/density (e.g. COUNT_CLASSES=strawberry).
# If unset and the model has "strawberry", only strawberry is counted.
# Otherwise totals exclude EXCLUDE_FROM_TOTAL classes (tobacco, soybean, etc.).
COUNT_CLASSES = os.getenv("COUNT_CLASSES", "").strip()
EXCLUDE_FROM_TOTAL = os.getenv("EXCLUDE_FROM_TOTAL", "tobacco,soybean")

valid_settings = {
    "runs_dir": os.environ.get("ULTRALYTICS_RUNS_DIR", "/tmp/runs"),
    "datasets_dir": "/tmp/datasets",
    "weights_dir": "/tmp/weights",
    "sync": False,
}
try:
    settings.update(valid_settings)
except KeyError:
    pass

# -------------------------
# Disk cache
# -------------------------
CACHE_DIR = os.getenv("BACKEND_CACHE_DIR", "/tmp/weedgeo_cache")
memory = Memory(CACHE_DIR, verbose=0)

# -------------------------
# FastAPI app / CORS
# -------------------------
app = FastAPI(title="Weed Geolocator API", version="1.2.0")

origins = os.getenv("CORS_ORIGINS", "http://localhost:8501,http://127.0.0.1:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in origins if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Utils
# -------------------------
def _file_fingerprint(path: str, algo: str = "sha1", block: int = 2**20) -> str:
    """Stable fingerprint so cache invalidates when file changes."""
    h = hashlib.new(algo)
    sz = os.path.getsize(path)
    mt = int(os.path.getmtime(path))
    h.update(str(sz).encode()); h.update(str(mt).encode())
    with open(path, "rb") as f:
        while True:
            chunk = f.read(block)
            if not chunk:
                break
            h.update(chunk)
    return f"{algo}:{h.hexdigest()}:{sz}:{mt}"

@lru_cache(maxsize=1)
def _get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"

@lru_cache(maxsize=256)
def _utm_epsg_for_lonlat(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return 32600 + zone if lat >= 0 else 32700 + zone

@lru_cache(maxsize=32)
def _get_transformer(from_crs: str | int, to_crs: str | int):
    return Transformer.from_crs(from_crs, to_crs, always_xy=True)

def save_upload_to_temp(uploaded: UploadFile, suffix: str = ".tif") -> str:
    with NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        uploaded.file.seek(0)
        shutil.copyfileobj(uploaded.file, tmp)
        return tmp.name

def counting_class_indices(class_names: List[str]) -> List[int]:
    """Which class indices contribute to total detections and density."""
    names = [str(n).strip().lower() for n in class_names]
    if COUNT_CLASSES:
        wanted = {c.strip().lower() for c in COUNT_CLASSES.split(",") if c.strip()}
        return [i for i, n in enumerate(names) if n in wanted]
    if "strawberry" in names:
        return [names.index("strawberry")]
    exclude = {c.strip().lower() for c in EXCLUDE_FROM_TOTAL.split(",") if c.strip()}
    return [i for i, n in enumerate(names) if n not in exclude]


def sum_counted_detections(class_names: List[str], totals_by_class: List[int]) -> int:
    idx = counting_class_indices(class_names)
    return int(sum(totals_by_class[i] for i in idx if 0 <= i < len(totals_by_class)))


def counted_class_names(class_names: List[str]) -> List[str]:
    idx = counting_class_indices(class_names)
    return [class_names[i] for i in idx if 0 <= i < len(class_names)]


def _stable_downsample_xy(x_list, y_list, max_keep=MAX_OVERLAY_POINTS_PER_CLASS):
    if max_keep <= 0 or len(x_list) <= max_keep:
        return x_list, y_list
    x = np.asarray(x_list, dtype=np.float64)
    y = np.asarray(y_list, dtype=np.float64)
    hv = np.floor(x * 1e3).astype(np.int64) * 6364136223846793005 ^ np.floor(y * 1e3).astype(np.int64)
    hv ^= (hv >> 33)
    order = np.argsort(hv)
    keep = order[:max_keep]
    return x[keep].tolist(), y[keep].tolist()


def _spatial_dedupe_keep_mask(
    x: np.ndarray, y: np.ndarray, conf: np.ndarray, radius_m: float
) -> np.ndarray:
    """Greedy NMS in plane: sort by descending conf; keep a point if no earlier kept point within radius_m."""
    n = int(x.size)
    if n == 0:
        return np.zeros(0, dtype=bool)
    radius_m = max(float(radius_m), 1e-6)
    r2 = radius_m * radius_m
    order = np.argsort(-conf.astype(np.float64))
    cell: dict[tuple[int, int], list[int]] = defaultdict(list)
    kept = np.zeros(n, dtype=bool)
    xv = x.astype(np.float64)
    yv = y.astype(np.float64)
    for o in order:
        xi = float(xv[o])
        yi = float(yv[o])
        ci, cj = int(np.floor(xi / radius_m)), int(np.floor(yi / radius_m))
        conflict = False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for pj in cell.get((ci + dx, cj + dy), ()):
                    dxm = xv[pj] - xi
                    dym = yv[pj] - yi
                    if dxm * dxm + dym * dym <= r2:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                break
        if not conflict:
            kept[o] = True
            cell[(ci, cj)].append(int(o))
    return kept


def apply_spatial_dedupe_to_inference_result(
    res: dict, radius_m: float, class_names: List[str]
) -> dict:
    """Spatial dedupe (UTM plane), rebuild per-class lists, lon/lat, and a dedup CSV (base64 in meta)."""
    ncls = len(class_names)
    xs, ys, cs, cfs, pxs, pys = [], [], [], [], [], []
    conf_by = res.get("conf_by_class") or [[] for _ in range(ncls)]
    for c in range(ncls):
        ux = res["utm_x_by_class"][c]
        uy = res["utm_y_by_class"][c]
        px = res["pix_x_by_class"][c]
        py = res["pix_y_by_class"][c]
        co = conf_by[c] if c < len(conf_by) else []
        for j in range(len(ux)):
            xs.append(ux[j])
            ys.append(uy[j])
            cs.append(c)
            cfs.append(float(co[j]) if j < len(co) else 1.0)
            pxs.append(px[j])
            pys.append(py[j])
    xa = np.asarray(xs, dtype=np.float64)
    ya = np.asarray(ys, dtype=np.float64)
    ca = np.asarray(cs, dtype=np.int64)
    cfa = np.asarray(cfs, dtype=np.float64)
    pxa = np.asarray(pxs, dtype=np.float64)
    pya = np.asarray(pys, dtype=np.float64)
    hdr = b"tile,class,confidence,xc_px,yc_px,w_px,h_px,global_x,global_y\n"
    if xa.size == 0:
        mm = dict(res["meta"])
        mm["dedup_csv_b64"] = base64.b64encode(hdr).decode("ascii")
        mm["dedup_radius_m"] = float(radius_m)
        mm["dedup_keep_count"] = 0
        out = dict(res)
        out["meta"] = mm
        return out
    mask = _spatial_dedupe_keep_mask(xa, ya, cfa, radius_m)
    epsg = int(res["meta"]["utm_epsg"])
    to_wgs84 = _get_transformer(f"EPSG:{epsg}", "EPSG:4326")
    new_utm_x = [[] for _ in range(ncls)]
    new_utm_y = [[] for _ in range(ncls)]
    new_pix_x = [[] for _ in range(ncls)]
    new_pix_y = [[] for _ in range(ncls)]
    new_lons = [[] for _ in range(ncls)]
    new_lats = [[] for _ in range(ncls)]
    new_conf = [[] for _ in range(ncls)]
    sbuf = io.StringIO()
    wr = csv_std.writer(sbuf)
    wr.writerow(["tile", "class", "confidence", "xc_px", "yc_px", "w_px", "h_px", "global_x", "global_y"])
    for i in np.flatnonzero(mask):
        c = int(ca[i])
        gx_f, gy_f = float(xa[i]), float(ya[i])
        new_utm_x[c].append(gx_f)
        new_utm_y[c].append(gy_f)
        new_pix_x[c].append(float(pxa[i]))
        new_pix_y[c].append(float(pya[i]))
        new_conf[c].append(float(cfa[i]))
        lon_f, lat_f = to_wgs84.transform(gx_f, gy_f)
        new_lons[c].append(float(lon_f))
        new_lats[c].append(float(lat_f))
        cls_name = class_names[c] if 0 <= c < len(class_names) else f"class_{c}"
        wr.writerow(
            [
                "mosaic",
                cls_name,
                f"{cfa[i]:.6f}",
                f"{pxa[i]:.1f}",
                f"{pya[i]:.1f}",
                "0",
                "0",
                gx_f,
                gy_f,
            ]
        )
    csv_b64 = base64.b64encode(sbuf.getvalue().encode("utf-8")).decode("ascii")
    totals_by_class = [len(new_utm_x[c]) for c in range(ncls)]
    total_weeds = sum_counted_detections(class_names, totals_by_class)
    fa = float(res["field_area_m2"])
    avg_density = (total_weeds / fa) if fa > 0 else 0.0
    out = dict(res)
    out["utm_x_by_class"] = new_utm_x
    out["utm_y_by_class"] = new_utm_y
    out["pix_x_by_class"] = new_pix_x
    out["pix_y_by_class"] = new_pix_y
    out["lons_by_class"] = new_lons
    out["lats_by_class"] = new_lats
    out["conf_by_class"] = new_conf
    out["totals_by_class"] = totals_by_class
    out["total_weeds"] = total_weeds
    out["avg_density"] = float(avg_density)
    mm = dict(res["meta"])
    mm["dedup_csv_b64"] = csv_b64
    mm["dedup_radius_m"] = float(radius_m)
    mm["dedup_keep_count"] = int(total_weeds)
    out["meta"] = mm
    return out


# -------------------------
# Model
# -------------------------
_MODEL = None
_DEVICE = None

def load_model(path: Path | str | None = None):
    """Load YOLO weights and sync CLASS_NAMES / NUM_CLASSES from the checkpoint (YOLOv8 or YOLOv12 fork)."""
    global _MODEL, _DEVICE, CLASS_NAMES, NUM_CLASSES
    if _MODEL is None:
        p = Path(path) if path is not None else MODEL_PATH
        p = p.expanduser().resolve()
        if not p.is_file():
            raise FileNotFoundError(f"Model weights not found: {p}")
        _MODEL = YOLO(str(p))
        # Match UI / inference loops to actual head output (e.g. single-class Strawberry).
        nd = getattr(_MODEL, "names", None)
        if isinstance(nd, dict):
            CLASS_NAMES = [nd[i] for i in sorted(nd.keys())]
        elif nd is not None:
            CLASS_NAMES = list(nd)
        if CLASS_NAMES:
            NUM_CLASSES = len(CLASS_NAMES)

        _DEVICE = _get_device()
        try:
            _MODEL.to(_DEVICE)
        except Exception:
            pass
        # Light warmup (single dummy) to initialize kernels
        dummy = np.zeros((TILE_SIZE, TILE_SIZE, 3), dtype=np.uint8)
        with torch.inference_mode():
            _MODEL.predict(
                dummy, imgsz=TILE_SIZE, conf=YOLO_CONF, iou=YOLO_IOU,
                verbose=False, device=_DEVICE
            )
    return _MODEL

# -------------------------
# Core inference (uncached body)
# -------------------------
def run_inference_from_path(src_path: str, tile_px: int = TILE_SIZE):
    model = load_model()

    with rasterio.open(src_path) as src:
        if src.crs is None:
            raise ValueError("Raster has no CRS; cannot compute physical sizes.")

        cx = (src.bounds.left + src.bounds.right) / 2.0
        cy = (src.bounds.bottom + src.bounds.top) / 2.0
        to_wgs84_src = _get_transformer(src.crs.to_string() if src.crs else "EPSG:4326", "EPSG:4326")
        lon_c, lat_c = to_wgs84_src.transform(cx, cy)
        utm_epsg = _utm_epsg_for_lonlat(lon_c, lat_c)

        with WarpedVRT(src, dst_crs=CRS.from_epsg(utm_epsg), resampling=Resampling.nearest) as r:
            width, height = r.width, r.height
            transform = r.transform
            crs = r.crs

            px_w = abs(transform.a)
            px_h = abs(transform.e)

            b = r.bounds
            utm_bounds = (b.left, b.bottom, b.right, b.top)
            to_wgs84_curr = _get_transformer(crs.to_string(), "EPSG:4326")

            nx = int(np.ceil(width / tile_px))
            ny = int(np.ceil(height / tile_px))

            totals_by_class = [0] * NUM_CLASSES
            pix_x_cls = [[] for _ in range(NUM_CLASSES)]
            pix_y_cls = [[] for _ in range(NUM_CLASSES)]
            utm_x_cls = [[] for _ in range(NUM_CLASSES)]
            utm_y_cls = [[] for _ in range(NUM_CLASSES)]
            conf_cls = [[] for _ in range(NUM_CLASSES)]
            lons_cls   = [[] for _ in range(NUM_CLASSES)]
            lats_cls   = [[] for _ in range(NUM_CLASSES)]

            bands_to_read = min(r.count, 3)
            batch_imgs, batch_info = [], []

            def flush_batch():
                if not batch_imgs:
                    return
                with torch.inference_mode():
                    results_list = model.predict(
                        batch_imgs,
                        imgsz=tile_px,
                        conf=YOLO_CONF,
                        iou=YOLO_IOU,
                        verbose=False,
                        device=_DEVICE,
                    )
                for res, (left_px, top_px) in zip(results_list, batch_info):
                    boxes = getattr(res, "boxes", None)
                    if boxes is None or boxes.xywh is None or len(boxes) == 0:
                        continue
                    xywh = np.array(boxes.xywh.tolist(), dtype=np.float32)
                    cls  = np.array(boxes.cls.tolist(),  dtype=np.int64)
                    confs = np.asarray(boxes.conf.detach().cpu().numpy(), dtype=np.float32).ravel()
                    if xywh.size == 0:
                        continue

                    cxs = xywh[:, 0]; cys = xywh[:, 1]
                    full_x = left_px + cxs
                    full_y = top_px + cys

                    xs_arr, ys_arr = r.xy(full_y, full_x)
                    xs_arr = np.atleast_1d(xs_arr).astype(np.float64).ravel()
                    ys_arr = np.atleast_1d(ys_arr).astype(np.float64).ravel()
                    lon, lat = to_wgs84_curr.transform(xs_arr, ys_arr)
                    lon = np.atleast_1d(lon).astype(np.float64).ravel()
                    lat = np.atleast_1d(lat).astype(np.float64).ravel()

                    for c in range(NUM_CLASSES):
                        m = (cls == c)
                        if not np.any(m):
                            continue
                        totals_by_class[c] += int(np.count_nonzero(m))
                        pix_x_cls[c].extend(full_x[m].tolist())
                        pix_y_cls[c].extend(full_y[m].tolist())
                        utm_x_cls[c].extend(xs_arr[m].tolist())
                        utm_y_cls[c].extend(ys_arr[m].tolist())
                        conf_cls[c].extend(confs[m].tolist())
                        ok = np.isfinite(lon[m]) & np.isfinite(lat[m])
                        if np.any(ok):
                            lons_cls[c].extend(lon[m][ok].tolist())
                            lats_cls[c].extend(lat[m][ok].tolist())

                batch_imgs.clear()
                batch_info.clear()

            for iy in range(ny):
                for ix in range(nx):
                    left_px = ix * tile_px
                    top_px  = iy * tile_px
                    right_px = min(left_px + tile_px, width)
                    bottom_px = min(top_px + tile_px, height)
                    win_w = right_px - left_px
                    win_h = bottom_px - top_px
                    if win_w <= 0 or win_h <= 0:
                        continue

                    # coarse mask to skip empties faster
                    msk = r.read_masks(
                        1,
                        window=Window(left_px, top_px, win_w, win_h),
                        out_shape=(max(1, win_h // 4), max(1, win_w // 4)),
                        resampling=Resampling.nearest,
                    )
                    if (msk == 0).mean() > 0.95:
                        continue

                    chip = r.read(
                        indexes=list(range(1, bands_to_read + 1)),
                        window=Window(left_px, top_px, win_w, win_h),
                        out_shape=(bands_to_read, win_h, win_w),
                        resampling=Resampling.nearest,
                    )
                    chip = np.transpose(chip, (1, 2, 0)).copy()
                    if chip.shape[2] == 1:
                        chip = np.repeat(chip, 3, axis=2)
                    elif chip.shape[2] > 3:
                        chip = chip[:, :, :3]
                    chip = np.ascontiguousarray(chip)
                    if chip.dtype != np.uint8:
                        chip = np.clip(chip, 0, 255).astype(np.uint8)

                    batch_imgs.append(chip)
                    batch_info.append((left_px, top_px))
                    if len(batch_imgs) >= BATCH_TILES:
                        flush_batch()
            flush_batch()

            field_area_m2 = float(width) * px_w * float(height) * px_h
            total_weeds = sum_counted_detections(CLASS_NAMES, totals_by_class)
            avg_density = (total_weeds / field_area_m2) if field_area_m2 > 0 else 0.0
            meta = {
                "width": int(width),
                "height": int(height),
                "px_w": float(px_w),
                "px_h": float(px_h),
                "utm_epsg": int(utm_epsg),
                "utm_bounds": tuple(float(x) for x in utm_bounds),
                "tile_size_px": int(tile_px),
                "transform": tuple(transform),
            }

    return {
        "totals_by_class": totals_by_class,
        "total_weeds": int(total_weeds),
        "field_area_m2": float(field_area_m2),
        "avg_density": float(avg_density),
        "pix_x_by_class": pix_x_cls,
        "pix_y_by_class": pix_y_cls,
        "utm_x_by_class": utm_x_cls,
        "utm_y_by_class": utm_y_cls,
        "conf_by_class": conf_cls,
        "lons_by_class":   lons_cls,
        "lats_by_class":   lats_cls,
        "meta": meta,
    }

# -------------------------
# Thumbnail & Grids
# -------------------------
def compute_thumbnail_from_path(src_path: str, max_display: int, utm_epsg: int):
    with rasterio.open(src_path) as src:
        with WarpedVRT(src, dst_crs=CRS.from_epsg(utm_epsg), resampling=Resampling.nearest) as ds:
            scale = min(max_display / ds.width, max_display / ds.height, 1.0)
            out_w = max(1, int(ds.width * scale))
            out_h = max(1, int(ds.height * scale))

            bands_to_read = min(ds.count, 3)
            thumb = ds.read(
                indexes=list(range(1, bands_to_read + 1)),
                out_shape=(bands_to_read, out_h, out_w),
                resampling=Resampling.nearest,
            )
            thumb = np.transpose(thumb, (1, 2, 0))
            if thumb.shape[2] == 1:
                thumb = np.repeat(thumb, 3, axis=2)

            img = Image.fromarray(np.clip(thumb, 0, 255).astype(np.uint8))
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            scale_x = out_w / ds.width
            scale_y = out_h / ds.height

    return png_bytes, (out_w, out_h), (scale_x, scale_y)

def build_counts_safe(utm_x, utm_y, utm_bounds, max_cells=MAX_CELLS):
    left, bottom, right, top = utm_bounds
    x0 = float(np.floor(left))
    y0 = float(np.floor(bottom))

    nx = int(np.ceil(right) - x0)
    ny = int(np.ceil(top) - y0)
    nx = max(nx, 1); ny = max(ny, 1)

    cell_m = 1
    if nx * ny > max_cells:
        cell_m = int(np.ceil(np.sqrt((nx * ny) / max_cells)))

    nx_c = max(1, int(np.ceil(nx / cell_m)))
    ny_c = max(1, int(np.ceil(ny / cell_m)))
    counts = np.zeros((ny_c, nx_c), dtype=np.uint32)

    if len(utm_x) == 0:
        return counts, x0, y0, cell_m

    utm_x = np.asarray(utm_x, dtype=np.float64)
    utm_y = np.asarray(utm_y, dtype=np.float64)

    ii = np.floor((utm_x - x0) / cell_m).astype(np.int64)
    jj = np.floor((utm_y - y0) / cell_m).astype(np.int64)
    ii = np.clip(ii, 0, nx_c - 1)
    jj = np.clip(jj, 0, ny_c - 1)

    np.add.at(counts, (jj, ii), 1)
    return counts, x0, y0, cell_m

def build_counts_fixed(utm_x, utm_y, utm_bounds, x0, y0, cell_m):
    left, bottom, right, top = utm_bounds
    nx = int(np.ceil(right) - x0)
    ny = int(np.ceil(top) - y0)
    nx_c = max(1, int(np.ceil(nx / cell_m)))
    ny_c = max(1, int(np.ceil(ny / cell_m)))
    counts = np.zeros((ny_c, nx_c), dtype=np.uint32)
    if len(utm_x) == 0:
        return counts
    utm_x = np.asarray(utm_x, dtype=np.float64)
    utm_y = np.asarray(utm_y, dtype=np.float64)
    ii = np.floor((utm_x - x0) / cell_m).astype(np.int64)
    jj = np.floor((utm_y - y0) / cell_m).astype(np.int64)
    ii = np.clip(ii, 0, counts.shape[1] - 1)
    jj = np.clip(jj, 0, counts.shape[0] - 1)
    np.add.at(counts, (jj, ii), 1)
    return counts

# -------------------------
# Shapefile builders
# -------------------------
def build_shapefile_zip_from_lonlat(lons, lats, add_class: bool = False, class_id: int = 0, class_name: str = "") -> bytes:
    tmp_dir = mkdtemp()
    shp_base = os.path.join(tmp_dir, "detections")
    out_zip_path = shp_base + ".zip"
    try:
        lx = np.asarray(lons, dtype=np.float64)
        ly = np.asarray(lats, dtype=np.float64)
        n = min(lx.size, ly.size)
        lx, ly = lx[:n], ly[:n]
        finite = np.isfinite(lx) & np.isfinite(ly)
        lx, ly = lx[finite], ly[finite]

        if lx.size == 0:
            empty_cols = {"id": []}
            if add_class:
                empty_cols.update({"class_id": [], "class_name": []})
            empty_gdf = gpd.GeoDataFrame(empty_cols, geometry=[], crs="EPSG:4326")
            geojson_path = shp_base + ".geojson"
            empty_gdf.to_file(geojson_path, driver="GeoJSON")
            shutil.make_archive(shp_base, "zip", tmp_dir, "detections.geojson")
        else:
            cols = {"id": np.arange(1, lx.size + 1)}
            if add_class:
                cols["class_id"] = [int(class_id)] * lx.size
                cols["class_name"] = [str(class_name)] * lx.size
            gdf = gpd.GeoDataFrame(
                cols,
                geometry=gpd.points_from_xy(lx, ly),
                crs="EPSG:4326",
            )
            try:
                gdf.to_file(filename=shp_base, driver="ESRI Shapefile", engine="pyogrio")
                shutil.make_archive(shp_base, "zip", tmp_dir, "detections")
            except Exception:
                geojson_path = shp_base + ".geojson"
                gdf.to_file(geojson_path, driver="GeoJSON")
                shutil.make_archive(shp_base, "zip", tmp_dir, "detections.geojson")

        with open(out_zip_path, "rb") as f:
            return f.read()
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

def build_shapefile_zip_all_and_per_class(lons_by_class: List[List[float]], lats_by_class: List[List[float]]) -> Tuple[bytes, List[bytes]]:
    all_lons = [x for xs in lons_by_class for x in xs]
    all_lats = [y for ys in lats_by_class for y in ys]
    all_zip = build_shapefile_zip_from_lonlat(all_lons, all_lats, add_class=False)

    per_class_zips = []
    for cid in range(len(lons_by_class)):
        z = build_shapefile_zip_from_lonlat(
            lons_by_class[cid], lats_by_class[cid],
            add_class=True, class_id=cid,
            class_name=(CLASS_NAMES[cid] if 0 <= cid < len(CLASS_NAMES) else f"class_{cid}")
        )
        per_class_zips.append(z)
    return all_zip, per_class_zips

# -------------------------
# Cached wrappers around heavy functions
# -------------------------
@memory.cache
def _run_inference_cached(src_path: str, tile_px: int, yolo_conf: float, yolo_iou: float,
                          class_names_tuple: tuple, file_sig: str):
    # Use globals for model thresholds during the call; restore after.
    global YOLO_CONF, YOLO_IOU
    old_conf, old_iou = YOLO_CONF, YOLO_IOU
    YOLO_CONF, YOLO_IOU = yolo_conf, yolo_iou
    try:
        return run_inference_from_path(src_path, tile_px)
    finally:
        YOLO_CONF, YOLO_IOU = old_conf, old_iou

@memory.cache
def _compute_thumbnail_cached(src_path: str, utm_epsg: int, max_display: int, file_sig: str):
    return compute_thumbnail_from_path(src_path, max_display, utm_epsg)

@memory.cache
def _build_shp_all_and_per_class_cached(lons_by_class, lats_by_class, class_names_tuple: tuple, file_sig: str):
    return build_shapefile_zip_all_and_per_class(lons_by_class, lats_by_class)

# -------------------------
# Schemas & Endpoints
# -------------------------
class InferResponse(BaseModel):
    total_weeds: int
    field_area_m2: float
    avg_density: float
    count_class_names: List[str] = []
    totals_by_class: List[int]
    class_names: List[str]

    meta: dict
    disp_w: int
    disp_h: int
    scale_x: float
    scale_y: float

    counts_base: List[List[int]]
    x0_m: float
    y0_m: float
    cell_m: int

    pix_x_by_class: List[List[float]]
    pix_y_by_class: List[List[float]]

    utm_x_by_class: List[List[float]]
    utm_y_by_class: List[List[float]]

    counts_base_by_class: List[List[List[int]]]

    thumb_png_b64: str
    shp_zip_b64_all: str
    shp_zip_b64_by_class: List[str]

@app.get("/health")
def health():
    _ = load_model()
    return {"status": "ok"}

@app.post("/infer", response_model=InferResponse)
def infer(
    file: UploadFile = File(...),
    apply_dedupe: bool = Form(False),
    dedupe_radius_m: float = Form(DEFAULT_DEDUPE_RADIUS_M),
):
    src_path = save_upload_to_temp(file)
    try:
        # Ensure classes come from the active checkpoint before cache key composition.
        _ = load_model()
        file_sig = _file_fingerprint(src_path)
        class_names_tuple = tuple(CLASS_NAMES)

        # ---------- 1) Inference (cached) ----------
        res = _run_inference_cached(
            src_path, TILE_SIZE, YOLO_CONF, YOLO_IOU, class_names_tuple, file_sig
        )
        if apply_dedupe:
            res = apply_spatial_dedupe_to_inference_result(
                res, float(dedupe_radius_m), list(class_names_tuple)
            )
        meta = res["meta"]
        shp_sig = f"{file_sig}|dedupe:{int(bool(apply_dedupe))}:{float(dedupe_radius_m):.4f}"

        # ---------- 2) Keep exactly the checkpoint classes ----------
        totals_by_class_kept = res["totals_by_class"]
        pix_x_by_class_kept  = res["pix_x_by_class"]
        pix_y_by_class_kept  = res["pix_y_by_class"]
        utm_x_by_class_kept  = res["utm_x_by_class"]
        utm_y_by_class_kept  = res["utm_y_by_class"]
        lons_by_class_kept   = res["lons_by_class"]
        lats_by_class_kept   = res["lats_by_class"]
        class_names_kept     = CLASS_NAMES

        total_weeds_kept = sum_counted_detections(class_names_kept, totals_by_class_kept)
        field_area_m2 = float(res["field_area_m2"])
        avg_density_kept = (total_weeds_kept / field_area_m2) if field_area_m2 > 0 else 0.0

        # ---------- 3) Thumbnail (cached) ----------
        thumb_png, (disp_w, disp_h), (scale_x, scale_y) = _compute_thumbnail_cached(
            src_path, meta["utm_epsg"], MAX_DISPLAY, file_sig
        )
        thumb_png_b64 = base64.b64encode(thumb_png).decode("utf-8")

        # ---------- 4) Shapefiles (cached) ----------
        shp_all, shp_by_cls = _build_shp_all_and_per_class_cached(
            lons_by_class_kept, lats_by_class_kept, tuple(class_names_kept), shp_sig
        )

        # ---------- 5) Combined base grid (fast) ----------
        all_utm_x = [x for xs in utm_x_by_class_kept for x in xs]
        all_utm_y = [y for ys in utm_y_by_class_kept for y in ys]
        if len(all_utm_x) == 0:
            left, bottom, right, top = meta["utm_bounds"]
            all_utm_x = [left, right]
            all_utm_y = [bottom, top]

        counts_base, x0_m, y0_m, cell_m = build_counts_safe(
            all_utm_x, all_utm_y, meta["utm_bounds"]
        )

        # ---------- 6) Per-class grids on same base ----------
        counts_base_by_class = []
        for i in range(len(utm_x_by_class_kept)):
            counts_c = build_counts_fixed(
                utm_x_by_class_kept[i], utm_y_by_class_kept[i],
                meta["utm_bounds"], x0_m, y0_m, cell_m
            )
            counts_base_by_class.append(counts_c.astype(int).tolist())

        # ---------- 7) Downsample overlay points for UI ----------
        pix_x_ds, pix_y_ds = [], []
        for i in range(len(pix_x_by_class_kept)):
            xds, yds = _stable_downsample_xy(pix_x_by_class_kept[i], pix_y_by_class_kept[i])
            pix_x_ds.append(xds)
            pix_y_ds.append(yds)

        payload = {
            "total_weeds": total_weeds_kept,
            "field_area_m2": field_area_m2,
            "avg_density": avg_density_kept,
            "count_class_names": counted_class_names(class_names_kept),
            "totals_by_class": totals_by_class_kept,
            "class_names": class_names_kept,

            "meta": meta,
            "disp_w": int(disp_w),
            "disp_h": int(disp_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),

            "counts_base": counts_base.astype(int).tolist(),
            "x0_m": float(x0_m),
            "y0_m": float(y0_m),
            "cell_m": int(cell_m),

            "pix_x_by_class": pix_x_ds,
            "pix_y_by_class": pix_y_ds,

            "utm_x_by_class": utm_x_by_class_kept,
            "utm_y_by_class": utm_y_by_class_kept,

            "counts_base_by_class": counts_base_by_class,

            "thumb_png_b64": thumb_png_b64,
            "shp_zip_b64_all": base64.b64encode(shp_all).decode("utf-8"),
            "shp_zip_b64_by_class": [base64.b64encode(z).decode("utf-8") for z in shp_by_cls],
        }
        return payload
    finally:
        try:
            os.unlink(src_path)
        except Exception:
            pass


@app.post("/infer_csv", response_model=InferResponse)
def infer_csv(file: UploadFile = File(...), detections_csv: UploadFile = File(...)):
    src_path = save_upload_to_temp(file, suffix=".tif")
    csv_path = save_upload_to_temp(detections_csv, suffix=".csv")
    try:
        file_sig = _file_fingerprint(src_path)
        csv_sig = _file_fingerprint(csv_path)
        combo_sig = f"{file_sig}|{csv_sig}"

        with rasterio.open(src_path) as src:
            if src.crs is None:
                raise ValueError("Raster has no CRS; cannot compute physical sizes.")

            cx = (src.bounds.left + src.bounds.right) / 2.0
            cy = (src.bounds.bottom + src.bounds.top) / 2.0
            to_wgs84_src = _get_transformer(src.crs.to_string(), "EPSG:4326")
            lon_c, lat_c = to_wgs84_src.transform(cx, cy)
            utm_epsg = _utm_epsg_for_lonlat(lon_c, lat_c)

            with WarpedVRT(src, dst_crs=CRS.from_epsg(utm_epsg), resampling=Resampling.nearest) as r:
                width, height = r.width, r.height
                transform = r.transform
                px_w = abs(transform.a)
                px_h = abs(transform.e)
                b = r.bounds
                utm_bounds = (b.left, b.bottom, b.right, b.top)
                to_wgs84_curr = _get_transformer(r.crs.to_string(), "EPSG:4326")

        df = pd.read_csv(csv_path)
        required = {"global_x", "global_y"}
        if not required.issubset(df.columns):
            raise ValueError("CSV must contain columns: global_x, global_y")

        if "class" in df.columns:
            class_vals = df["class"].fillna("plant").astype(str).str.strip()
            class_vals = class_vals.where(class_vals != "", "plant")
        else:
            class_vals = pd.Series(["plant"] * len(df))

        ordered_classes = list(dict.fromkeys(class_vals.tolist()))
        if not ordered_classes:
            ordered_classes = ["plant"]
        class_to_idx = {c: i for i, c in enumerate(ordered_classes)}
        ncls = len(ordered_classes)

        totals_by_class = [0] * ncls
        pix_x_by_class = [[] for _ in range(ncls)]
        pix_y_by_class = [[] for _ in range(ncls)]
        utm_x_by_class = [[] for _ in range(ncls)]
        utm_y_by_class = [[] for _ in range(ncls)]
        lons_by_class = [[] for _ in range(ncls)]
        lats_by_class = [[] for _ in range(ncls)]

        gx = pd.to_numeric(df["global_x"], errors="coerce").to_numpy(dtype=np.float64)
        gy = pd.to_numeric(df["global_y"], errors="coerce").to_numpy(dtype=np.float64)
        cls_arr = class_vals.to_numpy()
        finite = np.isfinite(gx) & np.isfinite(gy)
        gx = gx[finite]
        gy = gy[finite]
        cls_arr = cls_arr[finite]

        if gx.size:
            px, py = (~transform) * (gx, gy)
            px = np.asarray(px, dtype=np.float64)
            py = np.asarray(py, dtype=np.float64)
            in_bounds = (px >= 0) & (px < width) & (py >= 0) & (py < height)
            gx = gx[in_bounds]
            gy = gy[in_bounds]
            px = px[in_bounds]
            py = py[in_bounds]
            cls_arr = cls_arr[in_bounds]

            lon, lat = to_wgs84_curr.transform(gx, gy)
            lon = np.asarray(lon, dtype=np.float64)
            lat = np.asarray(lat, dtype=np.float64)

            for cls_name in ordered_classes:
                i = class_to_idx[cls_name]
                m = cls_arr == cls_name
                if not np.any(m):
                    continue
                totals_by_class[i] = int(np.count_nonzero(m))
                pix_x_by_class[i] = px[m].tolist()
                pix_y_by_class[i] = py[m].tolist()
                utm_x_by_class[i] = gx[m].tolist()
                utm_y_by_class[i] = gy[m].tolist()
                ok = np.isfinite(lon[m]) & np.isfinite(lat[m])
                if np.any(ok):
                    lons_by_class[i] = lon[m][ok].tolist()
                    lats_by_class[i] = lat[m][ok].tolist()

        field_area_m2 = float(width) * px_w * float(height) * px_h
        total_weeds = sum_counted_detections(ordered_classes, totals_by_class)
        avg_density = (total_weeds / field_area_m2) if field_area_m2 > 0 else 0.0
        meta = {
            "width": int(width),
            "height": int(height),
            "px_w": float(px_w),
            "px_h": float(px_h),
            "utm_epsg": int(utm_epsg),
            "utm_bounds": tuple(float(x) for x in utm_bounds),
            "tile_size_px": int(TILE_SIZE),
            "transform": tuple(transform),
            "source": "csv",
        }

        thumb_png, (disp_w, disp_h), (scale_x, scale_y) = _compute_thumbnail_cached(
            src_path, meta["utm_epsg"], MAX_DISPLAY, combo_sig
        )
        thumb_png_b64 = base64.b64encode(thumb_png).decode("utf-8")

        shp_all, shp_by_cls = _build_shp_all_and_per_class_cached(
            lons_by_class, lats_by_class, tuple(ordered_classes), combo_sig
        )

        all_utm_x = [x for xs in utm_x_by_class for x in xs]
        all_utm_y = [y for ys in utm_y_by_class for y in ys]
        if len(all_utm_x) == 0:
            left, bottom, right, top = meta["utm_bounds"]
            all_utm_x = [left, right]
            all_utm_y = [bottom, top]
        counts_base, x0_m, y0_m, cell_m = build_counts_safe(
            all_utm_x, all_utm_y, meta["utm_bounds"]
        )

        counts_base_by_class = []
        for i in range(len(utm_x_by_class)):
            counts_c = build_counts_fixed(
                utm_x_by_class[i], utm_y_by_class[i],
                meta["utm_bounds"], x0_m, y0_m, cell_m
            )
            counts_base_by_class.append(counts_c.astype(int).tolist())

        pix_x_ds, pix_y_ds = [], []
        for i in range(len(pix_x_by_class)):
            xds, yds = _stable_downsample_xy(pix_x_by_class[i], pix_y_by_class[i])
            pix_x_ds.append(xds)
            pix_y_ds.append(yds)

        return {
            "total_weeds": total_weeds,
            "field_area_m2": float(field_area_m2),
            "avg_density": float(avg_density),
            "count_class_names": counted_class_names(ordered_classes),
            "totals_by_class": totals_by_class,
            "class_names": ordered_classes,
            "meta": meta,
            "disp_w": int(disp_w),
            "disp_h": int(disp_h),
            "scale_x": float(scale_x),
            "scale_y": float(scale_y),
            "counts_base": counts_base.astype(int).tolist(),
            "x0_m": float(x0_m),
            "y0_m": float(y0_m),
            "cell_m": int(cell_m),
            "pix_x_by_class": pix_x_ds,
            "pix_y_by_class": pix_y_ds,
            "utm_x_by_class": utm_x_by_class,
            "utm_y_by_class": utm_y_by_class,
            "counts_base_by_class": counts_base_by_class,
            "thumb_png_b64": thumb_png_b64,
            "shp_zip_b64_all": base64.b64encode(shp_all).decode("utf-8"),
            "shp_zip_b64_by_class": [base64.b64encode(z).decode("utf-8") for z in shp_by_cls],
        }
    finally:
        for p in (src_path, csv_path):
            try:
                os.unlink(p)
            except Exception:
                pass
