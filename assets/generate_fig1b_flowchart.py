#!/usr/bin/env python3
"""Generate Fig. 1 field-counting flowchart (trained model assumed)."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon

OUT_DIR = Path(__file__).resolve().parent
DPI = 300

# Layout constants (figure coords 0–1)
COL_X = (0.16, 0.50, 0.84)   # Input, Processing, Output
BOX_W = 0.24
BOX_H = 0.052
ROW_STEP = 0.088
HDR_W = 0.26

C_IN = "#e8f5e9"
C_PROC = "#fff8e1"
C_OUT = "#e3f2fd"
C_HDR_IN, C_HDR_PROC, C_HDR_OUT = "#66bb6a", "#ffca28", "#42a5f5"
C_EDGE = "#37474f"
C_FILL = "#ffffff"
C_DONE = "#c8e6c9"


def rounded_box(ax, cx, cy, text, *, w=BOX_W, h=BOX_H, fc=C_FILL, ec=C_EDGE,
                fs=9, shape="rect", bold=False):
    x, y = cx - w / 2, cy - h / 2
    if shape == "diamond":
        pad = 0.006
        ax.add_patch(Polygon(
            [(cx, cy + h / 2 + pad), (cx + w / 2 + pad, cy),
             (cx, cy - h / 2 - pad), (cx - w / 2 - pad, cy)],
            closed=True, facecolor=fc, edgecolor=ec, lw=1.2, zorder=2))
    elif shape == "oval":
        ax.add_patch(mpatches.Ellipse(
            (cx, cy), w * 0.72, h * 0.92,
            facecolor=C_DONE, edgecolor="#2e7d32", lw=1.3, zorder=2))
    else:
        ax.add_patch(FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
            facecolor=fc, edgecolor=ec, lw=1.2, zorder=2))
    weight = "bold" if bold else "normal"
    ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
            fontfamily="sans-serif", weight=weight, zorder=3)


def column_header(ax, cx, y, label, color):
    ax.add_patch(FancyBboxPatch(
        (cx - HDR_W / 2, y - 0.017), HDR_W, 0.034,
        boxstyle="round,pad=0.01,rounding_size=0.01",
        facecolor=color, edgecolor=C_EDGE, lw=1.0, zorder=1))
    ax.text(cx, y, label, ha="center", va="center", fontsize=10,
            fontweight="bold", color="#212121", zorder=2)


def arrow_v(ax, x, y_top, y_bot, gap=0.006):
    ax.add_patch(FancyArrowPatch(
        (x, y_top - gap), (x, y_bot + gap),
        arrowstyle="-|>", mutation_scale=11, lw=1.1, color=C_EDGE,
        shrinkA=0, shrinkB=0, zorder=1))


def arrow_h(ax, x_left, x_right, y, gap=0.006):
    dx = 1 if x_right > x_left else -1
    ax.add_patch(FancyArrowPatch(
        (x_left + dx * gap, y), (x_right - dx * gap, y),
        arrowstyle="-|>", mutation_scale=11, lw=1.1, color=C_EDGE,
        shrinkA=0, shrinkB=0, zorder=1))


def draw_flowchart(ax):
    """Row-aligned 3-column flowchart — no overlapping boxes."""
    top_y = 0.90
    header_y = top_y - 0.04

    column_header(ax, COL_X[0], header_y, "Input", C_HDR_IN)
    column_header(ax, COL_X[1], header_y, "Processing", C_HDR_PROC)
    column_header(ax, COL_X[2], header_y, "Output", C_HDR_OUT)

    # Each row: (input, processing, output) — None = empty cell
    rows = [
        ("Georeferenced GeoTIFF\n(orthomosaic)", "Load orthomosaic", None),
        (None, "Reproject to UTM", None),
        ("Tile size, overlap\n& confidence", "Raster scan\n(overlapping tiles)", "Tile geotransform\n& offsets"),
        ("Trained YOLOv12\n(best.pt)", "YOLOv12 inference", None),
        (None, "Detections per tile", "global_x, global_y\n(from TIFF)"),
        ("DBSCAN parameters\n(eps, min_samples)", None, "DBSCAN deduplication"),
        (None, None, "Counts, heatmap\n& GIS export"),
        (None, None, "Results"),
    ]

    first_row_y = header_y - 0.075
    centers = {}  # (row, col) -> y

    for i, (inp, proc, out) in enumerate(rows):
        y = first_row_y - i * ROW_STEP
        if inp:
            rounded_box(ax, COL_X[0], y, inp, fc=C_IN)
            centers[(i, 0)] = y
        if proc:
            shape = "diamond" if "inference" in proc.lower() else "rect"
            rounded_box(ax, COL_X[1], y, proc, fc=C_PROC, shape=shape,
                        h=0.058 if shape == "diamond" else BOX_H)
            centers[(i, 1)] = y
        if out:
            shape = "oval" if out == "Results" else "rect"
            rounded_box(ax, COL_X[2], y, out, fc=C_OUT, shape=shape,
                        w=0.20 if shape == "oval" else BOX_W,
                        h=0.042 if shape == "oval" else BOX_H,
                        bold=(shape == "oval"))
            centers[(i, 2)] = y

    half_w = BOX_W / 2

    # Vertical flow — processing column
    proc_rows = sorted(r for (r, c) in centers if c == 1)
    for a, b in zip(proc_rows[:-1], proc_rows[1:]):
        arrow_v(ax, COL_X[1], centers[(a, 1)], centers[(b, 1)])

    # Vertical flow — output column
    out_rows = sorted(r for (r, c) in centers if c == 2)
    for a, b in zip(out_rows[:-1], out_rows[1:]):
        arrow_v(ax, COL_X[2], centers[(a, 2)], centers[(b, 2)])

    # Horizontal connectors (same row only — clean, no crossings)
    for i, (inp, proc, out) in enumerate(rows):
        if inp and proc and (i, 0) in centers and (i, 1) in centers:
            arrow_h(ax, COL_X[0] + half_w, COL_X[1] - half_w, centers[(i, 1)])
        if proc and out and (i, 1) in centers and (i, 2) in centers:
            arrow_h(ax, COL_X[1] + half_w, COL_X[2] - half_w, centers[(i, 2)])

    # Detections row → global coordinates (row 4): processing to output
    r_det = 4
    if (r_det, 1) in centers and (r_det, 2) in centers:
        arrow_h(ax, COL_X[1] + half_w, COL_X[2] - half_w, centers[(r_det, 2)])

    # DBSCAN params → deduplication (row 5)
    r_db = 5
    if (r_db, 0) in centers and (r_db, 2) in centers:
        arrow_h(ax, COL_X[0] + half_w, COL_X[2] - half_w, centers[(r_db, 2)])

    # Raster scan → tile geotransform (row 2)
    r_tile = 2
    if (r_tile, 1) in centers and (r_tile, 2) in centers:
        arrow_h(ax, COL_X[1] + half_w, COL_X[2] - half_w, centers[(r_tile, 2)])

    # Footnote
    foot_y = first_row_y - len(rows) * ROW_STEP - 0.02
    ax.text(
        0.50, foot_y,
        "Map coordinates derived from orthomosaic geotransform (not raw DJI EXIF GPS).",
        ha="center", va="top", fontsize=8.5, style="italic", color="#546e7a",
    )


def make_figure():
    fig, ax = plt.subplots(figsize=(10, 11))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.subplots_adjust(left=0.05, right=0.95, top=0.93, bottom=0.04)

    fig.suptitle(
        "Fig. 1. Field-scale plant counting from orthomosaic.",
        fontsize=12, fontweight="bold", y=0.97,
    )
    draw_flowchart(ax)
    return fig


def save_fig(fig, stem):
    for ext in ("png", "svg", "pdf"):
        p = OUT_DIR / f"{stem}.{ext}"
        fig.savefig(
            p, dpi=DPI if ext == "png" else None,
            bbox_inches="tight", pad_inches=0.25, facecolor="white",
        )
        print(f"Saved: {p}")


def main():
    for stem in ("fig1_methodology", "fig1b_processing_logic"):
        fig = make_figure()
        save_fig(fig, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()
