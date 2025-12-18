#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
mark_answers.py
Offline OMR-style bubble marking for standard multiple-choice answer sheets.

Supported input:
- PDF (single/multi-page) via PyMuPDF rendering for detection + direct PDF vector drawing for output
- High-res image (PNG/JPG) via OpenCV; output image or PDF

Constraints:
- No ML, no cloud; deterministic, modular, production-ready baseline
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import numpy as np
import cv2


# -----------------------------
# Answer key (question -> option 1..4)
# -----------------------------
ANSWER_KEY: Dict[int, int] = {
    1: 1,
    2: 4,
    3: 4,
    4: 1,
    5: 1,
    6: 3,
    7: 2,
    8: 1,
    9: 2,
    10: 3,
    11: 4,
    12: 2,
    13: 1,
    14: 3,
    15: 3,
    16: 4,
    17: 3,
    18: 3,
    19: 3,
    20: 4,
    21: 3,
    22: 4,
    23: 1,
    24: 3,
    25: 1,
    26: 4,
    27: 2,
    28: 4,
    29: 2,
    30: 3,
}


# -----------------------------
# Data structures
# -----------------------------
@dataclass(frozen=True)
class Bubble:
    bbox: Tuple[int, int, int, int]  # x, y, w, h in pixels
    center: Tuple[float, float]  # cx, cy in pixels
    area: float
    circularity: float


@dataclass(frozen=True)
class QuestionRow:
    question_number: int
    bubbles: List[Bubble]  # length 4, sorted left->right (options 1..4)


@dataclass
class Config:
    dpi: int = 400

    # Preprocessing
    blur_ksize: int = 5  # must be odd
    adaptive_block_size: int = 35  # must be odd
    adaptive_c: int = 8

    morph_open_ksize: int = 3
    morph_close_ksize: int = 3
    morph_open_iters: int = 1
    morph_close_iters: int = 1

    # Bubble filtering
    min_area: int = 300
    max_area: int = 20000
    aspect_min: float = 0.55
    aspect_max: float = 1.80
    extent_min: float = 0.35
    circularity_min: float = 0.35

    # Clustering / grouping
    x_cluster_thresh_factor: float = 0.75  # * median_diameter
    y_cluster_thresh_factor: float = 0.70  # * median_diameter
    column_gap_factor: float = 1.80  # * median_x_cluster_gap
    column_gap_min_diam_factor: float = 2.00

    # Marking
    inset_ratio: float = 0.18  # shrink bbox before filling, to preserve outline

    # Debug
    debug: bool = False
    debug_dir: Optional[str] = None


# -----------------------------
# Logging
# -----------------------------
def setup_logger(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


# -----------------------------
# Utilities
# -----------------------------
def ensure_odd(n: int, name: str) -> int:
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    logging.debug("%s set to odd value: %d", name, n)
    return n


def is_pdf(path: str) -> bool:
    return path.lower().endswith(".pdf")


def safe_makedirs(path: str) -> None:
    if path and not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def bgr_red() -> Tuple[int, int, int]:
    # Pure red in RGB is #FF0000 -> in OpenCV BGR: (0,0,255)
    return (0, 0, 255)


# -----------------------------
# Image I/O
# -----------------------------
def render_pdf_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = float(dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img_bgr


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


# -----------------------------
# Preprocessing
# -----------------------------
def preprocess_for_contours(img_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    k = ensure_odd(cfg.blur_ksize, "blur_ksize")
    if k > 1:
        gray = cv2.GaussianBlur(gray, (k, k), 0)

    block = ensure_odd(cfg.adaptive_block_size, "adaptive_block_size")
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        cfg.adaptive_c,
    )

    # Morph open to remove small noise
    if cfg.morph_open_ksize > 0:
        k_open = max(1, cfg.morph_open_ksize)
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        th = cv2.morphologyEx(
            th, cv2.MORPH_OPEN, kernel_open, iterations=cfg.morph_open_iters
        )

    # Morph close to connect small gaps in bubble edges
    if cfg.morph_close_ksize > 0:
        k_close = max(1, cfg.morph_close_ksize)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        th = cv2.morphologyEx(
            th, cv2.MORPH_CLOSE, kernel_close, iterations=cfg.morph_close_iters
        )

    return th


# -----------------------------
# Bubble detection
# -----------------------------
def contour_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * np.pi * area / (perimeter * perimeter))


def detect_bubbles(img_bgr: np.ndarray, cfg: Config) -> List[Bubble]:
    bin_img = preprocess_for_contours(img_bgr, cfg)
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bubbles: List[Bubble] = []
    for cnt in contours:
        area = float(cv2.contourArea(cnt))
        if area < cfg.min_area or area > cfg.max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w <= 1 or h <= 1:
            continue

        aspect = float(w) / float(h)
        if not (cfg.aspect_min <= aspect <= cfg.aspect_max):
            continue

        rect_area = float(w * h)
        extent = area / rect_area if rect_area > 0 else 0.0
        if extent < cfg.extent_min:
            continue

        perim = float(cv2.arcLength(cnt, True))
        circ = contour_circularity(area, perim)
        if circ < cfg.circularity_min:
            continue

        cx = float(x + w / 2.0)
        cy = float(y + h / 2.0)
        bubbles.append(
            Bubble(bbox=(x, y, w, h), center=(cx, cy), area=area, circularity=circ)
        )

    if not bubbles:
        logging.warning("No bubbles detected (after filtering).")
        return []

    logging.info("Detected %d candidate bubbles.", len(bubbles))
    return bubbles


# -----------------------------
# 1D clustering (deterministic, threshold-based)
# -----------------------------
def cluster_1d_sorted(
    values_with_ids: List[Tuple[float, int]], thresh: float
) -> List[List[int]]:
    """
    values_with_ids must be sorted by value asc.
    Returns clusters as lists of ids.
    """
    clusters: List[List[int]] = []
    if not values_with_ids:
        return clusters

    current: List[int] = [values_with_ids[0][1]]
    last_val = values_with_ids[0][0]

    for v, idx in values_with_ids[1:]:
        if abs(v - last_val) <= thresh:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
        last_val = v

    clusters.append(current)
    return clusters


# -----------------------------
# Layout grouping
# -----------------------------
def estimate_median_diameter(bubbles: List[Bubble]) -> float:
    ds = [0.5 * (b.bbox[2] + b.bbox[3]) for b in bubbles]
    return float(np.median(ds)) if ds else 0.0


def compute_cluster_means(
    bubbles: List[Bubble], clusters: List[List[int]], axis: str
) -> List[float]:
    means: List[float] = []
    for ids in clusters:
        vals = [
            (bubbles[i].center[0] if axis == "x" else bubbles[i].center[1]) for i in ids
        ]
        means.append(float(np.mean(vals)) if vals else 0.0)
    return means


def split_x_clusters_into_columns(
    x_clusters: List[List[int]], x_means: List[float], median_d: float, cfg: Config
) -> List[List[int]]:
    """
    x_clusters are option-position clusters (many across columns).
    We detect big gaps between consecutive x cluster means to split into column segments,
    then within each segment we chunk into groups of 4 (options 1..4).
    Returns list of columns, each as list of x_cluster indices (length 4 expected).
    """
    if len(x_clusters) < 4:
        logging.warning(
            "Not enough X clusters to form a column (need >=4, got %d).",
            len(x_clusters),
        )
        return []

    order = np.argsort(np.array(x_means)).tolist()
    x_means_sorted = [x_means[i] for i in order]

    gaps = [
        x_means_sorted[i + 1] - x_means_sorted[i]
        for i in range(len(x_means_sorted) - 1)
    ]
    if not gaps:
        return []

    median_gap = float(np.median(gaps)) if gaps else 0.0
    big_gap_thresh = max(
        cfg.column_gap_factor * median_gap, cfg.column_gap_min_diam_factor * median_d
    )

    boundaries: List[int] = []
    for i, g in enumerate(gaps):
        if g > big_gap_thresh:
            boundaries.append(i)

    # Create segments over sorted cluster indices
    segments: List[List[int]] = []
    start = 0
    for b in boundaries:
        seg = order[start : b + 1]
        if seg:
            segments.append(seg)
        start = b + 1
    tail = order[start:]
    if tail:
        segments.append(tail)

    # Within each segment, chunk into groups of 4 x-clusters (options 1..4)
    columns: List[List[int]] = []
    for seg in segments:
        seg_sorted = sorted(
            seg, key=lambda ci: x_means[ci]
        )  # left->right within segment
        if len(seg_sorted) < 4:
            logging.warning(
                "Segment too small for a column (size=%d). Skipping.", len(seg_sorted)
            )
            continue

        if len(seg_sorted) % 4 != 0:
            logging.warning(
                "X-cluster segment size %d is not a multiple of 4; will truncate remainder.",
                len(seg_sorted),
            )

        for k in range(0, len(seg_sorted) - (len(seg_sorted) % 4), 4):
            col = seg_sorted[k : k + 4]
            if len(col) == 4:
                columns.append(col)

    if not columns:
        logging.warning("Failed to split into valid columns.")
    else:
        logging.info("Inferred %d column(s).", len(columns))
    return columns


def assign_questions(
    bubbles: List[Bubble], cfg: Config, start_question_number: int = 1
) -> Tuple[List[QuestionRow], int, Dict[int, int], Dict[int, int]]:
    """
    Returns:
      - question rows (in numbering order)
      - next question number after this page
      - bubble_index -> x_cluster_index mapping
      - x_cluster_index -> column_index mapping
    """
    if not bubbles:
        return [], start_question_number, {}, {}

    median_d = estimate_median_diameter(bubbles)
    if median_d <= 0:
        logging.warning("Median diameter invalid; cannot group.")
        return [], start_question_number, {}, {}

    x_thresh = max(1.0, cfg.x_cluster_thresh_factor * median_d)
    y_thresh = max(1.0, cfg.y_cluster_thresh_factor * median_d)

    # Cluster by X into option-position peaks (across all columns)
    x_vals = sorted(
        [(bubbles[i].center[0], i) for i in range(len(bubbles))], key=lambda t: t[0]
    )
    x_clusters = cluster_1d_sorted(x_vals, x_thresh)
    x_means = compute_cluster_means(bubbles, x_clusters, "x")

    if len(x_clusters) < 4:
        logging.warning(
            "Too few X clusters (%d) for options; cannot assign questions.",
            len(x_clusters),
        )
        return [], start_question_number, {}, {}

    # Determine columns as groups of 4 x-clusters
    columns_xclusters = split_x_clusters_into_columns(
        x_clusters, x_means, median_d, cfg
    )
    if not columns_xclusters:
        return [], start_question_number, {}, {}

    # Map each x_cluster_index to column_index
    xcluster_to_col: Dict[int, int] = {}
    for col_idx, col in enumerate(
        sorted(columns_xclusters, key=lambda c: x_means[c[0]])
    ):
        for xc in col:
            xcluster_to_col[xc] = col_idx

    # Map bubble -> x_cluster_index (by membership)
    bubble_to_xcluster: Dict[int, int] = {}
    for xc_idx, ids in enumerate(x_clusters):
        for bi in ids:
            bubble_to_xcluster[bi] = xc_idx

    # Collect bubbles per column
    col_bubble_indices: List[List[int]] = [
        [] for _ in range(max(xcluster_to_col.values()) + 1)
    ]
    for bi in range(len(bubbles)):
        xc = bubble_to_xcluster.get(bi)
        if xc is None:
            continue
        col_idx = xcluster_to_col.get(xc)
        if col_idx is None:
            continue
        col_bubble_indices[col_idx].append(bi)

    # For each column, cluster by Y into rows (questions)
    question_rows: List[QuestionRow] = []
    qnum = start_question_number

    for col_idx in range(len(col_bubble_indices)):
        bis = col_bubble_indices[col_idx]
        if len(bis) < 4:
            logging.warning(
                "Column %d has too few bubbles (%d). Skipping.", col_idx, len(bis)
            )
            continue

        y_vals = sorted([(bubbles[i].center[1], i) for i in bis], key=lambda t: t[0])
        y_clusters = cluster_1d_sorted(y_vals, y_thresh)

        # Sort y-clusters by mean y (top->bottom)
        y_means = []
        for ids in y_clusters:
            y_means.append(
                float(np.mean([bubbles[i].center[1] for i in ids])) if ids else 0.0
            )
        y_order = sorted(range(len(y_clusters)), key=lambda k: y_means[k])

        for yi in y_order:
            row_ids = y_clusters[yi]
            if len(row_ids) != 4:
                logging.warning(
                    "Skipping invalid row in column %d (expected 4 bubbles, got %d).",
                    col_idx,
                    len(row_ids),
                )
                continue

            row_bubbles = [bubbles[i] for i in row_ids]
            row_bubbles.sort(key=lambda b: b.center[0])  # left->right options 1..4
            question_rows.append(QuestionRow(question_number=qnum, bubbles=row_bubbles))
            qnum += 1

    logging.info(
        "Assigned %d question row(s) starting at Q%d.",
        len(question_rows),
        start_question_number,
    )
    return question_rows, qnum, bubble_to_xcluster, xcluster_to_col


# -----------------------------
# PDF marking
# -----------------------------
def draw_filled_oval_pdf(
    page: fitz.Page, rect: fitz.Rect, fill_rgb: Tuple[float, float, float]
) -> None:
    """
    Draw a filled oval on the PDF page using a shape (more control than direct draw_oval).
    """
    shape = page.new_shape()
    shape.draw_oval(rect)
    shape.finish(color=None, fill=fill_rgb, width=0)
    shape.commit(overlay=True)


def mark_pdf_page(
    doc: fitz.Document,
    page_index: int,
    question_rows: List[QuestionRow],
    answer_key: Dict[int, int],
    render_w_px: int,
    render_h_px: int,
    cfg: Config,
) -> int:
    page = doc.load_page(page_index)
    pr = page.rect
    sx = pr.width / float(render_w_px)
    sy = pr.height / float(render_h_px)

    marked = 0
    for qr in question_rows:
        correct = answer_key.get(qr.question_number)
        if correct is None:
            continue
        if not (1 <= correct <= 4):
            continue

        bub = qr.bubbles[correct - 1]
        x, y, w, h = bub.bbox
        x0 = x * sx
        y0 = y * sy
        x1 = (x + w) * sx
        y1 = (y + h) * sy

        dx = (w * cfg.inset_ratio) * sx
        dy = (h * cfg.inset_ratio) * sy

        rect = fitz.Rect(x0 + dx, y0 + dy, x1 - dx, y1 - dy)
        # Fill pure red in RGB in PyMuPDF's 0..1 float space
        draw_filled_oval_pdf(page, rect, (1.0, 0.0, 0.0))
        marked += 1

    return marked


# -----------------------------
# Image marking
# -----------------------------
def mark_image_bgr(
    img_bgr: np.ndarray,
    question_rows: List[QuestionRow],
    answer_key: Dict[int, int],
    cfg: Config,
) -> int:
    marked = 0
    for qr in question_rows:
        correct = answer_key.get(qr.question_number)
        if correct is None:
            continue
        if not (1 <= correct <= 4):
            continue

        bub = qr.bubbles[correct - 1]
        x, y, w, h = bub.bbox

        # Inset ellipse to preserve outline
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        ax = int(round((w / 2.0) * (1.0 - cfg.inset_ratio)))
        ay = int(round((h / 2.0) * (1.0 - cfg.inset_ratio)))
        ax = max(1, ax)
        ay = max(1, ay)

        cv2.ellipse(
            img_bgr, (cx, cy), (ax, ay), 0.0, 0.0, 360.0, bgr_red(), thickness=-1
        )
        marked += 1

    return marked


# -----------------------------
# Debug visualization
# -----------------------------
def debug_render(
    img_bgr: np.ndarray, question_rows: List[QuestionRow], page_tag: str, cfg: Config
) -> None:
    if not cfg.debug:
        return

    out_dir = cfg.debug_dir or "debug_out"
    safe_makedirs(out_dir)

    dbg = img_bgr.copy()

    for qr in question_rows:
        # Draw row bbox and question number
        xs, ys, xes, yes = [], [], [], []
        for b in qr.bubbles:
            x, y, w, h = b.bbox
            xs.append(x)
            ys.append(y)
            xes.append(x + w)
            yes.append(y + h)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 1)

        x0 = min(xs) if xs else 0
        y0 = min(ys) if ys else 0
        x1 = max(xes) if xes else 0
        y1 = max(yes) if yes else 0
        cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 255, 0), 1)
        cv2.putText(
            dbg,
            f"Q{qr.question_number}",
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            dbg,
            f"Q{qr.question_number}",
            (x0, max(0, y0 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    out_path = os.path.join(out_dir, f"{page_tag}_debug.png")
    cv2.imwrite(out_path, dbg)
    logging.info("Debug image written: %s", out_path)


# -----------------------------
# Output helpers (image -> PDF)
# -----------------------------
def save_image_as_pdf(img_bgr: np.ndarray, output_pdf_path: str, dpi: int) -> None:
    h, w = img_bgr.shape[:2]
    width_pt = (w / float(dpi)) * 72.0
    height_pt = (h / float(dpi)) * 72.0

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_bytes = cv2.imencode(".png", img_rgb)[1].tobytes()

    doc = fitz.open()
    page = doc.new_page(width=width_pt, height=height_pt)
    rect = fitz.Rect(0, 0, width_pt, height_pt)
    page.insert_image(rect, stream=img_bytes, keep_proportion=False)
    doc.save(output_pdf_path, deflate=True, garbage=4)
    doc.close()


# -----------------------------
# Main processing
# -----------------------------
def process_pdf(input_path: str, output_path: str, cfg: Config) -> None:
    doc = fitz.open(input_path)
    qnum = 1
    total_marked = 0

    for pi in range(doc.page_count):
        page = doc.load_page(pi)
        img_bgr = render_pdf_page_to_bgr(page, cfg.dpi)
        h, w = img_bgr.shape[:2]

        bubbles = detect_bubbles(img_bgr, cfg)
        question_rows, qnum_next, _, _ = assign_questions(
            bubbles, cfg, start_question_number=qnum
        )

        debug_render(img_bgr, question_rows, page_tag=f"page_{pi+1}", cfg=cfg)

        marked = mark_pdf_page(
            doc=doc,
            page_index=pi,
            question_rows=question_rows,
            answer_key=ANSWER_KEY,
            render_w_px=w,
            render_h_px=h,
            cfg=cfg,
        )
        total_marked += marked
        qnum = qnum_next

        logging.info("Page %d/%d: marked %d bubble(s).", pi + 1, doc.page_count, marked)

    if total_marked == 0:
        logging.warning("No bubbles were marked. Check thresholds/layout assumptions.")

    doc.save(output_path, deflate=True, garbage=4)
    doc.close()
    logging.info("Output written: %s", output_path)


def process_image(input_path: str, output_path: str, cfg: Config) -> None:
    img_bgr = load_image_bgr(input_path)

    bubbles = detect_bubbles(img_bgr, cfg)
    question_rows, _, _, _ = assign_questions(bubbles, cfg, start_question_number=1)

    debug_render(img_bgr, question_rows, page_tag="image", cfg=cfg)

    marked = mark_image_bgr(img_bgr, question_rows, ANSWER_KEY, cfg)
    logging.info("Marked %d bubble(s).", marked)
    if marked == 0:
        logging.warning("No bubbles were marked. Check thresholds/layout assumptions.")

    if output_path.lower().endswith(".pdf"):
        save_image_as_pdf(img_bgr, output_path, cfg.dpi)
        logging.info("Output PDF written: %s", output_path)
    else:
        ok = cv2.imwrite(output_path, img_bgr)
        if not ok:
            raise ValueError(f"Failed to write image output: {output_path}")
        logging.info("Output image written: %s", output_path)


# -----------------------------
# CLI
# -----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mark_answers.py",
        description="Offline OMR bubble marker: fills correct option bubbles (pure red) based on a fixed answer key.",
    )
    p.add_argument("input", help="Input PDF or image (PNG/JPG).")
    p.add_argument("output", help="Output PDF (preferred) or image path.")
    p.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="Render DPI for PDF and assumed DPI for image->PDF. Default: 400",
    )

    # Preprocessing
    p.add_argument(
        "--blur-ksize",
        type=int,
        default=5,
        help="Gaussian blur kernel size (odd). Default: 5",
    )
    p.add_argument(
        "--block-size",
        type=int,
        default=35,
        help="Adaptive threshold block size (odd). Default: 35",
    )
    p.add_argument("--C", type=int, default=8, help="Adaptive threshold C. Default: 8")

    p.add_argument(
        "--morph-open-ksize",
        type=int,
        default=3,
        help="Morph open kernel size. Default: 3",
    )
    p.add_argument(
        "--morph-close-ksize",
        type=int,
        default=3,
        help="Morph close kernel size. Default: 3",
    )
    p.add_argument(
        "--morph-open-iters",
        type=int,
        default=1,
        help="Morph open iterations. Default: 1",
    )
    p.add_argument(
        "--morph-close-iters",
        type=int,
        default=1,
        help="Morph close iterations. Default: 1",
    )

    # Bubble filtering
    p.add_argument(
        "--min-area", type=int, default=300, help="Min contour area. Default: 300"
    )
    p.add_argument(
        "--max-area", type=int, default=20000, help="Max contour area. Default: 20000"
    )
    p.add_argument(
        "--aspect-min",
        type=float,
        default=0.55,
        help="Min bbox aspect ratio w/h. Default: 0.55",
    )
    p.add_argument(
        "--aspect-max",
        type=float,
        default=1.80,
        help="Max bbox aspect ratio w/h. Default: 1.80",
    )
    p.add_argument(
        "--extent-min",
        type=float,
        default=0.35,
        help="Min extent area/(w*h). Default: 0.35",
    )
    p.add_argument(
        "--circularity-min",
        type=float,
        default=0.35,
        help="Min circularity. Default: 0.35",
    )

    # Grouping
    p.add_argument(
        "--x-cluster-factor",
        type=float,
        default=0.75,
        help="X cluster thresh factor * median_d. Default: 0.75",
    )
    p.add_argument(
        "--y-cluster-factor",
        type=float,
        default=0.70,
        help="Y cluster thresh factor * median_d. Default: 0.70",
    )
    p.add_argument(
        "--column-gap-factor",
        type=float,
        default=1.80,
        help="Column split gap factor * median x-gap. Default: 1.80",
    )
    p.add_argument(
        "--column-gap-min-diam-factor",
        type=float,
        default=2.00,
        help="Min column split gap factor * median_d. Default: 2.00",
    )

    # Marking
    p.add_argument(
        "--inset-ratio",
        type=float,
        default=0.18,
        help="Inset ratio to preserve bubble outline. Default: 0.18",
    )

    # Debug
    p.add_argument(
        "--debug", action="store_true", help="Enable debug outputs (sidecar images)."
    )
    p.add_argument(
        "--debug-dir",
        default=None,
        help="Directory for debug outputs. Default: ./debug_out",
    )

    return p


def args_to_config(a: argparse.Namespace) -> Config:
    return Config(
        dpi=int(a.dpi),
        blur_ksize=int(a.blur_ksize),
        adaptive_block_size=int(a.block_size),
        adaptive_c=int(a.C),
        morph_open_ksize=int(a.morph_open_ksize),
        morph_close_ksize=int(a.morph_close_ksize),
        morph_open_iters=int(a.morph_open_iters),
        morph_close_iters=int(a.morph_close_iters),
        min_area=int(a.min_area),
        max_area=int(a.max_area),
        aspect_min=float(a.aspect_min),
        aspect_max=float(a.aspect_max),
        extent_min=float(a.extent_min),
        circularity_min=float(a.circularity_min),
        x_cluster_thresh_factor=float(a.x_cluster_factor),
        y_cluster_thresh_factor=float(a.y_cluster_factor),
        column_gap_factor=float(a.column_gap_factor),
        column_gap_min_diam_factor=float(a.column_gap_min_diam_factor),
        inset_ratio=float(a.inset_ratio),
        debug=bool(a.debug),
        debug_dir=str(a.debug_dir) if a.debug_dir else None,
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = args_to_config(args)
    setup_logger(cfg.debug)

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        logging.error("Input not found: %s", input_path)
        return 2

    # Prepare debug dir if needed
    if cfg.debug:
        cfg.debug_dir = cfg.debug_dir or "debug_out"
        safe_makedirs(cfg.debug_dir)

    try:
        if is_pdf(input_path):
            if not output_path.lower().endswith(".pdf"):
                logging.warning(
                    "Input is PDF; output is not PDF. Continuing, but PDF output is recommended."
                )
            process_pdf(input_path, output_path, cfg)
        else:
            process_image(input_path, output_path, cfg)
    except Exception as e:
        logging.exception("Failed: %s", str(e))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
