# 2.1 Methodology — Field-Scale Plant Counting

This document describes the **field-counting workflow** shown in Fig. 1. A **pre-trained YOLOv12 model** (`best.pt`) is applied to a **georeferenced orthomosaic GeoTIFF** to detect plants and map each detection to field coordinates.

**Figure:** `fig1_methodology.png` — Fig. 1 field-counting flowchart

Regenerate: `python assets/generate_fig1b_flowchart.py`

---

## 2.1.1 Workflow Overview (Fig. 1)

Fig. 1 is organised in three columns — **Input**, **Processing**, and **Output** — with each row representing one stage of the pipeline.

| Step | Input | Processing | Output |
|------|-------|------------|--------|
| 1 | Georeferenced GeoTIFF (orthomosaic) | Load orthomosaic | — |
| 2 | — | Reproject to UTM | — |
| 3 | Tile size, overlap & confidence | Raster scan (overlapping tiles) | Tile geotransform & offsets |
| 4 | Trained YOLOv12 (`best.pt`) | YOLOv12 inference | — |
| 5 | — | Detections per tile | `global_x`, `global_y` (from TIFF) |
| 6 | DBSCAN parameters (`eps`, `min_samples`) | — | DBSCAN deduplication |
| 7 | — | — | Counts, heatmap & GIS export |
| 8 | — | — | Results |

```
Georeferenced GeoTIFF
        │
        ▼
Load orthomosaic ──► Reproject to UTM
        │
        ▼
Raster scan (overlapping tiles) ──► Tile geotransform & offsets
        │
        ▼
YOLOv12 inference (best.pt) ──► Detections per tile
        │
        ▼
global_x, global_y (from TIFF geotransform)
        │
        ▼
DBSCAN deduplication ──► Counts, heatmap & GIS export ──► Results
```

Map coordinates are derived from the **orthomosaic geotransform**, not from raw DJI EXIF GPS.

---

## 2.1.2 Step 1 — Load Orthomosaic

**Input:** A georeferenced GeoTIFF orthomosaic covering the full field.

The orthomosaic is produced upstream by photogrammetry software (e.g. Pix4D, OpenDroneMap) from UAV flight images. It is the sole raster input to the counting pipeline.

**Processing:** The GeoTIFF is opened with Rasterio. The embedded coordinate reference system (CRS) and affine geotransform are read from the file metadata.

---

## 2.1.3 Step 2 — Reproject to UTM

**Processing:** The orthomosaic is reprojected to a UTM coordinate system for consistent spatial measurements across the field.

In the web application (`plant_counting_ui`), this is handled on-the-fly via `WarpedVRT` during windowed reads. In the notebook pipeline (`geotagging_refactored.ipynb`), tiles retain their individual geotransforms after cropping.

---

## 2.1.4 Step 3 — Raster Scan

**Input:** Tile size, overlap fraction, and detection confidence threshold.

**Processing:** The orthomosaic is too large for a single GPU pass. It is read in **overlapping windows** (a raster scan). Each window is passed to the detector independently.

| Implementation | Tile size | Overlap |
|----------------|-----------|---------|
| Notebook (`geotagging_refactored.ipynb`) | 1600 × 1300 px | 70% |
| Web UI (`plant_counting_ui`) | 640 × 640 px (default) | Configurable |

**Output:** For each tile, the **pixel offset** (`x0`, `y0`) and **geotransform** are recorded so detections can be mapped back to the full raster.

Tiles may be saved as intermediate GeoTIFFs in the notebook pipeline; each intermediate file retains its own geotransform.

---

## 2.1.5 Step 4 — YOLOv12 Inference

**Input:** Pre-trained YOLOv12 weights (`best.pt`).

**Processing:** The model runs on each tile. Tiles are resized to **1280 × 1280** pixels during `predict()` to match the training resolution. Bounding boxes and class labels are returned per tile.

---

## 2.1.6 Step 5 — Detections per Tile → Global Coordinates

**Processing:** For every detection in a tile:

1. Compute the bounding-box centre in **tile-local pixels** `(xc, yc)`.
2. Add the tile offset to obtain **full orthomosaic pixel** position.
3. Apply the raster **geotransform** to obtain map coordinates.

**Output:** Each detection is assigned **`global_x`** and **`global_y`** and written to `global_detections.csv`.

```python
# geotagging_refactored.ipynb
gx, gy = transform * (xc, yc)   # → global_x, global_y

# plant_counting_ui backend
full_x = left_px + detection_center_x
utm_x, utm_y = raster.xy(full_y, full_x)
```

| Field | Description |
|-------|-------------|
| `global_x`, `global_y` | Map coordinates from **TIFF geotransform** |

---

## 2.1.7 Step 6 — Deduplication

**Input:** Deduplication parameters (method depends on implementation — see §2.2.5 for the web tool).

**Notebook pipeline:** Overlapping tiles can detect the same plant more than once. Duplicates are removed with **DBSCAN** (`sklearn.cluster.DBSCAN`, `uav/dbscan_deduplication.py`):

1. Build feature vectors from `global_x`, `global_y` (optionally `confidence`, `class`).
2. Standardise with `StandardScaler`.
3. Cluster with `eps` and `min_samples`.
4. Per cluster, **keep the highest-confidence detection**.

**Output:** `deduplicated_detections_dbscan.csv`.

---

## 2.1.8 Step 7 — Counts, Heatmap & GIS Export

**Output:**

- Per-class plant counts and field density (plants m⁻²)
- Detection map and density heatmap
- GIS export (shapefile, CSV)

---

## 2.1.9 Why the Orthomosaic, Not Raw RGB Images

| | Raw RGB (`DJI_*.JPG`) | Stitched GeoTIFF |
|---|----------------------|------------------|
| Coverage | Separate snapshots, each with its own perspective | One aligned, georectified raster |
| Coordinates | Camera GPS in EXIF only — not per-pixel | CRS + affine geotransform for every pixel |
| Use in counting | Not suitable for field-scale mapping | Required for `global_x`, `global_y` |

---

## 2.1.10 Software and Libraries

| Component | Technology |
|-----------|------------|
| Object detection | YOLOv12 (pre-trained), PyTorch |
| Orthomosaic I/O | Rasterio, PyProj |
| Deduplication (notebook) | scikit-learn (DBSCAN) |
| Deduplication (web tool) | Spatial radius merge (UTM plane) |
| Web UI / API | Streamlit, FastAPI, GeoPandas |

---

## 2.1.11 Figure Caption

**Fig. 1.** Schematic of field-scale plant counting from orthomosaic. A georeferenced GeoTIFF is loaded and reprojected to UTM, raster-scanned in overlapping tiles, and processed with a pre-trained YOLOv12 model. Tile geotransforms map detections to `global_x` and `global_y` coordinates. DBSCAN removes duplicates arising from tile overlap. Final outputs include plant counts, density heatmaps, and GIS exports.

---

## 2.1.12 Suggested Manuscript Paragraph

> For field-scale plant counting, a georeferenced orthomosaic GeoTIFF was loaded and reprojected to UTM. The raster was scanned in overlapping tiles and processed with a pre-trained YOLOv12 detector. Detection coordinates were derived from the orthomosaic geotransform (`global_x`, `global_y`), using tile offsets to map local detections to the full field. DBSCAN clustering removed duplicate detections in overlap regions, retaining the highest-confidence detection per cluster. Final outputs included per-class plant counts, density heatmaps, and GIS-compatible exports.

---

## 2.1.13 Parameter Summary

| Parameter | Value |
|-----------|-------|
| Raster input | Georeferenced GeoTIFF (orthomosaic) |
| Model | Pre-trained YOLOv12 (`best.pt`) |
| Model input size | 1280 × 1280 px (via YOLO `imgsz`) |
| Inference tile size | 1600 × 1300 px (notebook) or 640 × 640 px (web UI) |
| Inference tile overlap | 70% (notebook); non-overlapping grid (web UI, 640 px windows) |
| Confidence threshold | 0.25 default (web UI backend); user-defined in notebook |
| Coordinates | `global_x`, `global_y` from TIFF geotransform |
| Deduplication (notebook) | DBSCAN (`eps`, `min_samples`) |
| Deduplication (web UI) | Radius-based merge (default 0.5 m, user-adjustable) |

---

## 2.2 Web-Based Plant Counting Tool

A browser-based application (`plant_counting_ui`) wraps the field-counting pipeline in an interactive interface for non-programmatic use. It implements the same core stages as Fig. 1 — load orthomosaic, reproject to UTM, raster scan, YOLO inference, coordinate assignment, deduplication, and export — while adding visual exploration and one-click downloads.

### 2.2.1 Architecture

The tool uses a **client–server** design:

| Layer | Technology | Role |
|-------|------------|------|
| Frontend | Streamlit (`frontend/app.py`) | File upload, parameter controls, maps, charts, downloads |
| Backend | FastAPI (`backend/main.py`) | Tiled inference, coordinate conversion, deduplication, GIS packaging |
| Model | Ultralytics YOLO (`best.pt`) | Loaded at API startup via `MODEL_PATH` environment variable |

The frontend communicates with the backend over HTTP (`POST /infer`, `POST /infer_csv`, `GET /health`). Both services can be run locally or deployed together via Docker Compose (API on port 8000, UI on port 8501).

### 2.2.2 User Workflow

The sidebar guides users through five steps:

1. **Upload** a georeferenced GeoTIFF (`.tif` / `.tiff`).
2. **Choose a data source** — run YOLO with optional deduplication, run YOLO without deduplication, or visualise an existing detections CSV.
3. **Run detection** — sends the orthomosaic to the API and caches results for the session.
4. **Explore** detection overlays and density heatmaps on an orthomosaic thumbnail.
5. **Export** deduplicated CSV and per-class shapefile archives.

### 2.2.3 Data Source Modes

| Mode | Description |
|------|-------------|
| **Run YOLO on orthomosaic + dedupe + CSV** | Full pipeline: tiled inference, spatial deduplication, CSV export, and map visualisation. User sets dedupe radius (default 0.5 m). |
| **Run YOLO on orthomosaic (raw, no dedupe)** | Inference only; all tile-level detections retained. Useful for inspecting raw model output. |
| **Plot detections from uploaded CSV** | Skips YOLO. User uploads a CSV with `global_x` and `global_y` columns (e.g. from the notebook pipeline); the tool maps points onto the orthomosaic and computes counts and heatmaps. |

### 2.2.4 Backend Processing

When a GeoTIFF is submitted to `POST /infer`, the API:

1. Opens the raster with Rasterio and selects an appropriate **UTM EPSG** from the orthomosaic centre.
2. Reads the image in **640 × 640 pixel windows** (`TILE_SIZE`, non-overlapping grid) via `WarpedVRT` for on-the-fly UTM reprojection.
3. Runs **batched YOLO inference** (default confidence 0.25, IoU 0.45) on each tile.
4. Converts each detection centre to **UTM coordinates** (`global_x`, `global_y`) using the tile window offset and raster geotransform.
5. Optionally applies **spatial deduplication** — detections within a user-specified radius on the UTM plane are merged, keeping the highest-confidence point per cluster.
6. Computes **field area** (m²) and **average density** (detections m⁻²).
7. Generates a downsampled **orthomosaic thumbnail** for display, **per-class count grids**, and **shapefile ZIP archives** (all classes combined and one per class).

Results are disk-cached on the backend (`joblib.Memory`) and session-cached on the frontend (`st.cache_data`) so repeated views of the same upload are fast.

### 2.2.5 Deduplication in the Web Tool

The web tool uses a **radius-based spatial merge** in UTM coordinates (`apply_spatial_dedupe_to_inference_result` in `backend/main.py`), not DBSCAN. This is simpler to tune interactively: the user sets a single **dedupe radius in metres** (0.05–25 m). Detections closer than this distance are grouped; the highest-confidence detection in each group is kept.

The research notebook pipeline (§2.1.7) uses **DBSCAN** instead. Both methods address the same problem — duplicate detections from tiled inference — but serve different contexts: reproducible batch analysis (notebook) versus interactive parameter tuning (web tool).

### 2.2.6 Visualisation and Metrics

After a successful run, the dashboard displays:

**Summary metrics (top row)**

- Total detections
- Field area (m²)
- Average density (detections m⁻²)
- Dominant class

**Per-class breakdown**

- Bar chart and table of detection counts per class
- Colour-coded class chips

**Detections map tab**

- Orthomosaic thumbnail with detection points overlaid
- Class filter (multiselect) and adjustable marker size
- Interactive Plotly map (pan/zoom disabled for stable overlay)

**Density heatmap tab**

- Detection density grid overlaid on the orthomosaic
- Scope: all classes or a single class
- Adjustable cell size (metres) and heatmap opacity
- Colour scale: yellow–orange–red (detections per cell)

**Run metadata**

- Model class names, base grid cell size, and source filename

### 2.2.7 Export

The **Export** tab provides one-click downloads:

| File | Contents |
|------|----------|
| `deduplicated_detections.csv` | Columns: `tile`, `class`, `confidence`, `xc_px`, `yc_px`, `w_px`, `h_px`, `global_x`, `global_y` (available when dedupe mode is used) |
| `detections_all.zip` | Shapefile of all detections (WGS84 lon/lat) |
| `detections_<class>.zip` | Per-class shapefile archives |

Shapefiles are generated server-side with GeoPandas and returned as base64-encoded ZIP blobs.

### 2.2.8 Configuration

Key environment variables for deployment:

| Variable | Default | Purpose |
|----------|---------|---------|
| `MODEL_PATH` | `models/bigmodel.pt` | Path to YOLO weights |
| `TILE_SIZE` | `640` | Inference window size (pixels) |
| `YOLO_CONF` | `0.25` | Detection confidence threshold |
| `YOLO_IOU` | `0.45` | NMS IoU threshold |
| `DEDUPE_RADIUS_M` | `0.5` | Default dedupe radius (metres) |
| `API_BASE` | `http://127.0.0.1:8000` | Backend URL (frontend) |

### 2.2.9 Suggested Manuscript Paragraph (Web Tool)

> To make field-scale counting accessible without command-line tools, we developed a web-based plant counting application comprising a FastAPI inference service and a Streamlit front end. Users upload a georeferenced orthomosaic GeoTIFF, select an analysis mode, and run tiled YOLOv12 inference with optional spatial deduplication. The interface displays per-class counts, field density, detection overlays, and adjustable density heatmaps on an orthomosaic preview. Results can be exported as CSV and GIS shapefiles for downstream analysis. The tool implements the same orthomosaic-based coordinate workflow as the batch notebook pipeline while providing interactive visualisation and parameter control.

---

*Field counting from orthomosaic GeoTIFF: load → reproject → raster scan → YOLO inference → geotransform coordinates → deduplication → counts & GIS export. Interactive access via the Plant Counting web tool (§2.2).*
