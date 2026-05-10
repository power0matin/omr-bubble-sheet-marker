#!/usr/bin/env python3
"""
Offline deterministic OMR bubble marker.

The pipeline is intentionally non-ML and production-oriented:
- PDF pages are rendered with PyMuPDF for detection and marked with vector overlays.
- Images are processed with OpenCV and can be saved as images or print-quality PDFs.
- Bubble detection uses contour geometry plus global size-consistency filtering, which
  is much more robust than accepting every circle-like contour. This prevents question
  numbers, option labels, and decorative text from being treated as answer bubbles.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import asdict, dataclass
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import cv2
import fitz
import numpy as np

# -----------------------------------------------------------------------------
# Built-in answer key (question -> option 1..4). Can be overridden with JSON.
# -----------------------------------------------------------------------------
ANSWER_KEY: dict[int, int] = {
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


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class Bubble:
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    area: float
    circularity: float
    extent: float
    aspect: float

    @property
    def diameter(self) -> float:
        _, _, w, h = self.bbox
        return 0.5 * float(w + h)


@dataclass(frozen=True, slots=True)
class QuestionRow:
    question_number: int
    bubbles: list[Bubble]


@dataclass(frozen=True, slots=True)
class DetectionReport:
    raw_contours: int = 0
    geometric_candidates: int = 0
    size_filtered_candidates: int = 0
    deduped_bubbles: int = 0
    selected_median_diameter: float = 0.0
    selected_size_cluster_count: int = 0
    x_cluster_count: int = 0
    inferred_column_count: int = 0
    assigned_question_count: int = 0
    skipped_row_count: int = 0

    def with_updates(self, **kwargs: Any) -> DetectionReport:
        values = asdict(self)
        values.update(kwargs)
        return DetectionReport(**values)


@dataclass(slots=True)
class Config:
    dpi: int = 400
    expected_options: int = 4

    # Preprocessing
    blur_ksize: int = 5
    adaptive_block_size: int = 35
    adaptive_c: int = 8
    morph_open_ksize: int = 3
    morph_close_ksize: int = 3
    morph_open_iters: int = 1
    morph_close_iters: int = 1

    # Initial contour geometry filters
    min_area: int = 300
    max_area: int = 20000
    aspect_min: float = 0.55
    aspect_max: float = 1.80
    extent_min: float = 0.35
    circularity_min: float = 0.35

    # Robust global bubble-size filtering. This is the main protection against
    # text/labels being marked as bubbles.
    auto_size_filter: bool = True
    size_cluster_tolerance: float = 0.18
    duplicate_center_factor: float = 0.35

    # Clustering / grouping
    x_cluster_thresh_factor: float = 0.75
    y_cluster_thresh_factor: float = 0.70
    column_gap_factor: float = 1.80
    column_gap_min_diam_factor: float = 2.00

    # Marking
    inset_ratio: float = 0.18

    # Debug
    debug: bool = False
    debug_dir: str | None = None


# -----------------------------------------------------------------------------
# Logging and validation
# -----------------------------------------------------------------------------
def setup_logger(debug: bool) -> None:
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s", force=True)


def ensure_odd(n: int, name: str) -> int:
    if n < 3:
        n = 3
    if n % 2 == 0:
        n += 1
    logging.debug("%s set to odd value: %d", name, n)
    return n


def normalize_config(cfg: Config) -> Config:
    cfg.dpi = max(72, int(cfg.dpi))
    cfg.expected_options = max(2, int(cfg.expected_options))
    cfg.blur_ksize = ensure_odd(int(cfg.blur_ksize), "blur_ksize")
    cfg.adaptive_block_size = ensure_odd(int(cfg.adaptive_block_size), "adaptive_block_size")
    cfg.morph_open_ksize = max(0, int(cfg.morph_open_ksize))
    cfg.morph_close_ksize = max(0, int(cfg.morph_close_ksize))
    cfg.morph_open_iters = max(0, int(cfg.morph_open_iters))
    cfg.morph_close_iters = max(0, int(cfg.morph_close_iters))
    cfg.min_area = max(1, int(cfg.min_area))
    cfg.max_area = max(cfg.min_area + 1, int(cfg.max_area))
    cfg.size_cluster_tolerance = min(max(float(cfg.size_cluster_tolerance), 0.05), 0.75)
    cfg.duplicate_center_factor = min(max(float(cfg.duplicate_center_factor), 0.05), 1.50)
    cfg.inset_ratio = min(max(float(cfg.inset_ratio), 0.0), 0.45)
    return cfg


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------
def is_pdf(path: str | os.PathLike[str]) -> bool:
    return str(path).lower().endswith(".pdf")


def safe_makedirs(path: str | os.PathLike[str] | None) -> None:
    if path:
        Path(path).mkdir(parents=True, exist_ok=True)


def bgr_red() -> tuple[int, int, int]:
    return (0, 0, 255)


def contour_circularity(area: float, perimeter: float) -> float:
    if perimeter <= 1e-6:
        return 0.0
    return float(4.0 * math.pi * area / (perimeter * perimeter))


def median_or_zero(values: Sequence[float]) -> float:
    return float(np.median(np.asarray(values, dtype=np.float64))) if values else 0.0


# -----------------------------------------------------------------------------
# Answer key loading
# -----------------------------------------------------------------------------
def load_answer_key(path: str | None, expected_options: int = 4) -> dict[int, int]:
    """Load an answer key from JSON or return the built-in key.

    Supported JSON formats:
    - {"1": 2, "2": 4, ...}
    - [2, 4, 1, ...]  # list index 0 means question 1
    """
    if not path:
        return dict(ANSWER_KEY)

    key_path = Path(path)
    with key_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        answer_key = {i + 1: int(v) for i, v in enumerate(data)}
    elif isinstance(data, dict):
        answer_key = {int(k): int(v) for k, v in data.items()}
    else:
        raise ValueError("Answer key JSON must be either an object or a list.")

    for q, option in answer_key.items():
        if q < 1:
            raise ValueError(f"Invalid question number in answer key: {q}")
        if not (1 <= option <= expected_options):
            raise ValueError(f"Invalid answer for Q{q}: {option}; expected 1..{expected_options}.")

    logging.info("Loaded %d answer(s) from %s", len(answer_key), key_path)
    return answer_key


# -----------------------------------------------------------------------------
# Image I/O
# -----------------------------------------------------------------------------
def render_pdf_page_to_bgr(page: fitz.Page, dpi: int) -> np.ndarray:
    zoom = float(dpi) / 72.0
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)  # RGB bytes
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def load_image_bgr(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {path}")
    return img


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------
def preprocess_for_contours(img_bgr: np.ndarray, cfg: Config) -> np.ndarray:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if cfg.blur_ksize > 1:
        gray = cv2.GaussianBlur(gray, (cfg.blur_ksize, cfg.blur_ksize), 0)

    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        cfg.adaptive_block_size,
        cfg.adaptive_c,
    )

    if cfg.morph_open_ksize > 0 and cfg.morph_open_iters > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_open_ksize, cfg.morph_open_ksize)
        )
        th = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations=cfg.morph_open_iters)

    if cfg.morph_close_ksize > 0 and cfg.morph_close_iters > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.morph_close_ksize, cfg.morph_close_ksize)
        )
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=cfg.morph_close_iters)

    return th


# -----------------------------------------------------------------------------
# Bubble detection
# -----------------------------------------------------------------------------
def contour_to_bubble(cnt: np.ndarray, cfg: Config) -> Bubble | None:
    area = float(cv2.contourArea(cnt))
    if area < cfg.min_area or area > cfg.max_area:
        return None

    x, y, w, h = cv2.boundingRect(cnt)
    if w <= 1 or h <= 1:
        return None

    aspect = float(w) / float(h)
    if not (cfg.aspect_min <= aspect <= cfg.aspect_max):
        return None

    rect_area = float(w * h)
    extent = area / rect_area if rect_area > 0 else 0.0
    if extent < cfg.extent_min:
        return None

    perimeter = float(cv2.arcLength(cnt, True))
    circularity = contour_circularity(area, perimeter)
    if circularity < cfg.circularity_min:
        return None

    return Bubble(
        bbox=(int(x), int(y), int(w), int(h)),
        center=(float(x + w / 2.0), float(y + h / 2.0)),
        area=area,
        circularity=circularity,
        extent=extent,
        aspect=aspect,
    )


def cluster_diameters(candidates: list[Bubble], tolerance: float) -> list[list[Bubble]]:
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda b: b.diameter)
    clusters: list[list[Bubble]] = [[ordered[0]]]

    for bubble in ordered[1:]:
        current = clusters[-1]
        current_median = median_or_zero([b.diameter for b in current])
        allowed_gap = max(3.0, tolerance * max(current_median, 1.0))
        if abs(bubble.diameter - current_median) <= allowed_gap:
            current.append(bubble)
        else:
            clusters.append([bubble])

    return clusters


def select_dominant_size_cluster(
    candidates: list[Bubble], cfg: Config
) -> tuple[list[Bubble], float, int]:
    """Pick the most plausible bubble-size cluster.

    Text often passes circularity/extent filters, but it is not globally consistent with the
    true bubble size. We therefore choose the largest size-consistent group. If two groups
    have the same count (outer and inner contours of hollow bubbles), we keep the larger one.
    """
    if not candidates or not cfg.auto_size_filter:
        return candidates, median_or_zero([b.diameter for b in candidates]), len(candidates)

    clusters = cluster_diameters(candidates, cfg.size_cluster_tolerance)
    if not clusters:
        return [], 0.0, 0

    viable_clusters = [
        cluster for cluster in clusters if len(cluster) >= max(cfg.expected_options * 3, 8)
    ] or clusters

    max_count = max(len(cluster) for cluster in viable_clusters)
    near_max_tolerance = max(cfg.expected_options, int(round(0.10 * max_count)))
    near_max_clusters = [
        cluster for cluster in viable_clusters if (max_count - len(cluster)) <= near_max_tolerance
    ]

    # If inner and outer contours both exist, their counts are usually identical or
    # very close. Prefer the larger diameter cluster to keep the outer bubble boundary.
    best = max(near_max_clusters, key=lambda cluster: median_or_zero([b.diameter for b in cluster]))
    median_d = median_or_zero([b.diameter for b in best])
    if median_d <= 0:
        return [], 0.0, 0

    # Return the selected cluster itself. Re-expanding around the median can accidentally
    # pull inner contours or option-label digits back into the candidate set.
    return list(best), median_d, len(best)


def dedupe_near_identical_bubbles(
    candidates: list[Bubble], median_diameter: float, cfg: Config
) -> list[Bubble]:
    if not candidates:
        return []

    threshold = max(3.0, cfg.duplicate_center_factor * max(median_diameter, 1.0))
    selected: list[Bubble] = []

    # Larger area first keeps outer contours over inner contours when both survive.
    for cand in sorted(candidates, key=lambda b: b.area, reverse=True):
        cx, cy = cand.center
        duplicate = False
        for kept in selected:
            kx, ky = kept.center
            if math.hypot(cx - kx, cy - ky) <= threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(cand)

    selected.sort(key=lambda b: (b.center[1], b.center[0]))
    return selected


def detect_bubbles_with_report(
    img_bgr: np.ndarray, cfg: Config
) -> tuple[list[Bubble], DetectionReport]:
    binary = preprocess_for_contours(img_bgr, cfg)

    # RETR_LIST is intentional: it still sees bubbles when the sheet has a large page border
    # around all content. RETR_EXTERNAL would only return the page border in that case.
    contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    report = DetectionReport(raw_contours=len(contours))

    geometric: list[Bubble] = []
    for cnt in contours:
        bubble = contour_to_bubble(cnt, cfg)
        if bubble is not None:
            geometric.append(bubble)

    report = report.with_updates(geometric_candidates=len(geometric))
    if not geometric:
        logging.warning("No bubble-like contours detected after geometry filtering.")
        return [], report

    size_filtered, median_d, cluster_count = select_dominant_size_cluster(geometric, cfg)
    deduped = dedupe_near_identical_bubbles(size_filtered, median_d, cfg)

    report = report.with_updates(
        size_filtered_candidates=len(size_filtered),
        deduped_bubbles=len(deduped),
        selected_median_diameter=round(median_d, 3),
        selected_size_cluster_count=cluster_count,
    )

    if not deduped:
        logging.warning("No bubbles detected after size filtering/deduplication.")
    else:
        logging.info(
            "Detected %d bubble(s): raw=%d, geometric=%d, size_filtered=%d, median_d=%.2f",
            len(deduped),
            report.raw_contours,
            report.geometric_candidates,
            report.size_filtered_candidates,
            median_d,
        )

    return deduped, report


def detect_bubbles(img_bgr: np.ndarray, cfg: Config) -> list[Bubble]:
    bubbles, _ = detect_bubbles_with_report(img_bgr, cfg)
    return bubbles


# -----------------------------------------------------------------------------
# 1D clustering and layout grouping
# -----------------------------------------------------------------------------
def cluster_1d_sorted(values_with_ids: list[tuple[float, int]], thresh: float) -> list[list[int]]:
    if not values_with_ids:
        return []

    clusters: list[list[int]] = []
    current: list[int] = [values_with_ids[0][1]]
    current_values: list[float] = [values_with_ids[0][0]]

    for value, idx in values_with_ids[1:]:
        center = float(np.mean(current_values))
        if abs(value - center) <= thresh:
            current.append(idx)
            current_values.append(value)
        else:
            clusters.append(current)
            current = [idx]
            current_values = [value]

    clusters.append(current)
    return clusters


def estimate_median_diameter(bubbles: list[Bubble]) -> float:
    return median_or_zero([b.diameter for b in bubbles])


def compute_cluster_means(
    bubbles: list[Bubble], clusters: list[list[int]], axis: str
) -> list[float]:
    means: list[float] = []
    for ids in clusters:
        vals = [(bubbles[i].center[0] if axis == "x" else bubbles[i].center[1]) for i in ids]
        means.append(median_or_zero(vals))
    return means


def split_x_clusters_into_columns(
    x_clusters: list[list[int]], x_means: list[float], median_d: float, cfg: Config
) -> list[list[int]]:
    expected = cfg.expected_options
    if len(x_clusters) < expected:
        logging.warning(
            "Not enough X clusters to form a question group (need >=%d, got %d).",
            expected,
            len(x_clusters),
        )
        return []

    order = np.argsort(np.asarray(x_means, dtype=np.float64)).tolist()
    x_means_sorted = [x_means[i] for i in order]
    gaps = [x_means_sorted[i + 1] - x_means_sorted[i] for i in range(len(x_means_sorted) - 1)]

    if not gaps:
        return [order] if len(order) == expected else []

    median_gap = median_or_zero(gaps)
    big_gap_threshold = max(
        cfg.column_gap_factor * median_gap, cfg.column_gap_min_diam_factor * median_d
    )
    boundaries = [i for i, gap in enumerate(gaps) if gap > big_gap_threshold]

    segments: list[list[int]] = []
    start = 0
    for boundary in boundaries:
        segment = order[start : boundary + 1]
        if segment:
            segments.append(segment)
        start = boundary + 1
    tail = order[start:]
    if tail:
        segments.append(tail)

    columns: list[list[int]] = []
    for segment in segments:
        segment_sorted = sorted(segment, key=lambda ci: x_means[ci])
        if len(segment_sorted) < expected:
            logging.warning(
                "Skipping X segment with %d cluster(s); need %d.", len(segment_sorted), expected
            )
            continue
        if len(segment_sorted) % expected != 0:
            logging.warning(
                "X segment has %d clusters, not a multiple of %d; truncating remainder.",
                len(segment_sorted),
                expected,
            )
        usable = len(segment_sorted) - (len(segment_sorted) % expected)
        for i in range(0, usable, expected):
            column = segment_sorted[i : i + expected]
            if len(column) == expected:
                columns.append(column)

    columns.sort(key=lambda col: x_means[col[0]])
    logging.info("Inferred %d column(s).", len(columns))
    return columns


def assign_questions(
    bubbles: list[Bubble], cfg: Config, start_question_number: int = 1
) -> tuple[list[QuestionRow], int, dict[int, int], dict[int, int], DetectionReport]:
    if not bubbles:
        return [], start_question_number, {}, {}, DetectionReport()

    median_d = estimate_median_diameter(bubbles)
    if median_d <= 0:
        logging.warning("Median bubble diameter is invalid; cannot group.")
        return [], start_question_number, {}, {}, DetectionReport()

    x_thresh = max(1.0, cfg.x_cluster_thresh_factor * median_d)
    y_thresh = max(1.0, cfg.y_cluster_thresh_factor * median_d)

    x_values = sorted(
        [(bubbles[i].center[0], i) for i in range(len(bubbles))], key=lambda item: item[0]
    )
    x_clusters = cluster_1d_sorted(x_values, x_thresh)
    x_means = compute_cluster_means(bubbles, x_clusters, "x")

    columns_xclusters = split_x_clusters_into_columns(x_clusters, x_means, median_d, cfg)
    if not columns_xclusters:
        return [], start_question_number, {}, {}, DetectionReport(x_cluster_count=len(x_clusters))

    xcluster_to_col: dict[int, int] = {}
    for col_idx, column in enumerate(columns_xclusters):
        for xcluster in column:
            xcluster_to_col[xcluster] = col_idx

    bubble_to_xcluster: dict[int, int] = {}
    for xcluster_idx, ids in enumerate(x_clusters):
        for bubble_idx in ids:
            bubble_to_xcluster[bubble_idx] = xcluster_idx

    col_bubble_indices: list[list[int]] = [[] for _ in range(len(columns_xclusters))]
    for bubble_idx in range(len(bubbles)):
        xcluster = bubble_to_xcluster.get(bubble_idx)
        col_idx = xcluster_to_col.get(xcluster) if xcluster is not None else None
        if col_idx is not None:
            col_bubble_indices[col_idx].append(bubble_idx)

    question_rows: list[QuestionRow] = []
    skipped_rows = 0
    qnum = start_question_number

    for col_idx, bubble_indices in enumerate(col_bubble_indices):
        if len(bubble_indices) < cfg.expected_options:
            logging.warning("Column %d has too few bubbles (%d).", col_idx + 1, len(bubble_indices))
            continue

        y_values = sorted(
            [(bubbles[i].center[1], i) for i in bubble_indices], key=lambda item: item[0]
        )
        y_clusters = cluster_1d_sorted(y_values, y_thresh)
        y_means = [median_or_zero([bubbles[i].center[1] for i in ids]) for ids in y_clusters]
        y_order = sorted(range(len(y_clusters)), key=lambda idx: y_means[idx])

        for y_idx in y_order:
            row_ids = y_clusters[y_idx]
            if len(row_ids) != cfg.expected_options:
                skipped_rows += 1
                logging.warning(
                    "Skipping row in column %d: expected %d bubbles, got %d.",
                    col_idx + 1,
                    cfg.expected_options,
                    len(row_ids),
                )
                continue

            row_bubbles = sorted([bubbles[i] for i in row_ids], key=lambda b: b.center[0])
            question_rows.append(QuestionRow(question_number=qnum, bubbles=row_bubbles))
            qnum += 1

    logging.info(
        "Assigned %d question row(s) starting at Q%d.", len(question_rows), start_question_number
    )

    report = DetectionReport(
        x_cluster_count=len(x_clusters),
        inferred_column_count=len(columns_xclusters),
        assigned_question_count=len(question_rows),
        skipped_row_count=skipped_rows,
    )
    return question_rows, qnum, bubble_to_xcluster, xcluster_to_col, report


# -----------------------------------------------------------------------------
# Marking
# -----------------------------------------------------------------------------
def draw_filled_oval_pdf(
    page: fitz.Page, rect: fitz.Rect, fill_rgb: tuple[float, float, float]
) -> None:
    shape = page.new_shape()
    shape.draw_oval(rect)
    shape.finish(color=None, fill=fill_rgb, width=0)
    shape.commit(overlay=True)


def mark_pdf_page(
    doc: fitz.Document,
    page_index: int,
    question_rows: list[QuestionRow],
    answer_key: dict[int, int],
    render_w_px: int,
    render_h_px: int,
    cfg: Config,
) -> int:
    page = doc.load_page(page_index)
    page_rect = page.rect
    sx = page_rect.width / float(render_w_px)
    sy = page_rect.height / float(render_h_px)

    marked = 0
    for row in question_rows:
        correct = answer_key.get(row.question_number)
        if correct is None or not (1 <= correct <= len(row.bubbles)):
            continue

        x, y, w, h = row.bubbles[correct - 1].bbox
        dx = (w * cfg.inset_ratio) * sx
        dy = (h * cfg.inset_ratio) * sy
        rect = fitz.Rect(x * sx + dx, y * sy + dy, (x + w) * sx - dx, (y + h) * sy - dy)
        draw_filled_oval_pdf(page, rect, (1.0, 0.0, 0.0))
        marked += 1

    return marked


def mark_image_bgr(
    img_bgr: np.ndarray,
    question_rows: list[QuestionRow],
    answer_key: dict[int, int],
    cfg: Config,
) -> int:
    marked = 0
    for row in question_rows:
        correct = answer_key.get(row.question_number)
        if correct is None or not (1 <= correct <= len(row.bubbles)):
            continue

        x, y, w, h = row.bubbles[correct - 1].bbox
        cx = int(round(x + w / 2.0))
        cy = int(round(y + h / 2.0))
        ax = max(1, int(round((w / 2.0) * (1.0 - cfg.inset_ratio))))
        ay = max(1, int(round((h / 2.0) * (1.0 - cfg.inset_ratio))))
        cv2.ellipse(img_bgr, (cx, cy), (ax, ay), 0.0, 0.0, 360.0, bgr_red(), thickness=-1)
        marked += 1

    return marked


# -----------------------------------------------------------------------------
# Debug outputs
# -----------------------------------------------------------------------------
def merge_reports(*reports: DetectionReport) -> DetectionReport:
    values = asdict(DetectionReport())
    for report in reports:
        for key, value in asdict(report).items():
            if isinstance(value, (int, float)) and value != 0:
                values[key] = value
    return DetectionReport(**values)


def debug_render(
    img_bgr: np.ndarray,
    bubbles: list[Bubble],
    question_rows: list[QuestionRow],
    page_tag: str,
    cfg: Config,
    report: DetectionReport | None = None,
) -> None:
    if not cfg.debug:
        return

    out_dir = cfg.debug_dir or "debug_out"
    safe_makedirs(out_dir)

    dbg = img_bgr.copy()

    for bubble in bubbles:
        x, y, w, h = bubble.bbox
        cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 180, 0), 1)

    for row in question_rows:
        xs, ys, xes, yes = [], [], [], []
        for bubble in row.bubbles:
            x, y, w, h = bubble.bbox
            xs.append(x)
            ys.append(y)
            xes.append(x + w)
            yes.append(y + h)
            cv2.rectangle(dbg, (x, y), (x + w, y + h), (255, 0, 0), 1)

        if xs:
            x0, y0, x1, y1 = min(xs), min(ys), max(xes), max(yes)
            cv2.rectangle(dbg, (x0, y0), (x1, y1), (0, 180, 0), 2)
            cv2.putText(
                dbg,
                f"Q{row.question_number}",
                (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 0, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                dbg,
                f"Q{row.question_number}",
                (x0, max(0, y0 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )

    image_path = os.path.join(out_dir, f"{page_tag}_debug.png")
    cv2.imwrite(image_path, dbg)
    logging.info("Debug image written: %s", image_path)

    if report is not None:
        report_path = os.path.join(out_dir, f"{page_tag}_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(asdict(report), f, indent=2, ensure_ascii=False)
        logging.info("Debug report written: %s", report_path)


# -----------------------------------------------------------------------------
# Output helpers
# -----------------------------------------------------------------------------
def save_image_as_pdf(img_bgr: np.ndarray, output_pdf_path: str, dpi: int) -> None:
    h, w = img_bgr.shape[:2]
    width_pt = (w / float(dpi)) * 72.0
    height_pt = (h / float(dpi)) * 72.0

    # Important: cv2.imencode expects BGR. Do not convert to RGB here, otherwise
    # the red marks become blue in the generated PDF.
    ok, encoded = cv2.imencode(".png", img_bgr)
    if not ok:
        raise ValueError("Failed to encode image as PNG for PDF output.")

    doc = fitz.open()
    page = doc.new_page(width=width_pt, height=height_pt)
    page.insert_image(
        fitz.Rect(0, 0, width_pt, height_pt), stream=encoded.tobytes(), keep_proportion=False
    )
    doc.save(output_pdf_path, deflate=True, garbage=4)
    doc.close()


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------
def process_pdf(input_path: str, output_path: str, cfg: Config, answer_key: dict[int, int]) -> None:
    doc = fitz.open(input_path)
    qnum = 1
    total_marked = 0

    for page_index in range(doc.page_count):
        page = doc.load_page(page_index)
        img_bgr = render_pdf_page_to_bgr(page, cfg.dpi)
        h, w = img_bgr.shape[:2]

        bubbles, detection_report = detect_bubbles_with_report(img_bgr, cfg)
        question_rows, qnum_next, _, _, layout_report = assign_questions(
            bubbles, cfg, start_question_number=qnum
        )
        report = merge_reports(detection_report, layout_report)

        debug_render(
            img_bgr,
            bubbles,
            question_rows,
            page_tag=f"page_{page_index + 1}",
            cfg=cfg,
            report=report,
        )

        marked = mark_pdf_page(
            doc=doc,
            page_index=page_index,
            question_rows=question_rows,
            answer_key=answer_key,
            render_w_px=w,
            render_h_px=h,
            cfg=cfg,
        )
        total_marked += marked
        qnum = qnum_next
        logging.info("Page %d/%d: marked %d bubble(s).", page_index + 1, doc.page_count, marked)

    if total_marked == 0:
        logging.warning("No bubbles were marked. Check thresholds/layout assumptions.")

    doc.save(output_path, deflate=True, garbage=4)
    doc.close()
    logging.info("Output written: %s", output_path)


def process_image(
    input_path: str, output_path: str, cfg: Config, answer_key: dict[int, int]
) -> None:
    img_bgr = load_image_bgr(input_path)

    bubbles, detection_report = detect_bubbles_with_report(img_bgr, cfg)
    question_rows, _, _, _, layout_report = assign_questions(bubbles, cfg, start_question_number=1)
    report = merge_reports(detection_report, layout_report)

    debug_render(img_bgr, bubbles, question_rows, page_tag="image", cfg=cfg, report=report)

    marked = mark_image_bgr(img_bgr, question_rows, answer_key, cfg)
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


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omr-mark",
        description="Offline OMR bubble marker: fills correct option bubbles in pure red.",
    )
    parser.add_argument("input", help="Input PDF or image path (PNG/JPG/JPEG).")
    parser.add_argument("output", help="Output PDF or image path.")
    parser.add_argument(
        "--dpi", type=int, default=400, help="PDF render/image PDF DPI. Default: 400"
    )
    parser.add_argument(
        "--answer-key",
        default=None,
        help="Optional JSON answer key. Supports {'1': 2} or [2, 4, ...].",
    )
    parser.add_argument(
        "--expected-options",
        type=int,
        default=4,
        help="Number of bubbles per question row. Default: 4",
    )

    parser.add_argument(
        "--blur-ksize", type=int, default=5, help="Gaussian blur kernel size. Default: 5"
    )
    parser.add_argument(
        "--block-size", type=int, default=35, help="Adaptive threshold block size. Default: 35"
    )
    parser.add_argument("--C", type=int, default=8, help="Adaptive threshold C. Default: 8")
    parser.add_argument(
        "--morph-open-ksize", type=int, default=3, help="Morph open kernel. Default: 3"
    )
    parser.add_argument(
        "--morph-close-ksize", type=int, default=3, help="Morph close kernel. Default: 3"
    )
    parser.add_argument(
        "--morph-open-iters", type=int, default=1, help="Morph open iterations. Default: 1"
    )
    parser.add_argument(
        "--morph-close-iters", type=int, default=1, help="Morph close iterations. Default: 1"
    )

    parser.add_argument(
        "--min-area", type=int, default=300, help="Minimum contour area. Default: 300"
    )
    parser.add_argument(
        "--max-area", type=int, default=20000, help="Maximum contour area. Default: 20000"
    )
    parser.add_argument(
        "--aspect-min", type=float, default=0.55, help="Minimum width/height. Default: 0.55"
    )
    parser.add_argument(
        "--aspect-max", type=float, default=1.80, help="Maximum width/height. Default: 1.80"
    )
    parser.add_argument(
        "--extent-min", type=float, default=0.35, help="Minimum contour extent. Default: 0.35"
    )
    parser.add_argument(
        "--circularity-min", type=float, default=0.35, help="Minimum circularity. Default: 0.35"
    )
    parser.add_argument(
        "--no-auto-size-filter",
        action="store_true",
        help="Disable dominant bubble-size filtering. Not recommended for real sheets.",
    )
    parser.add_argument(
        "--size-cluster-tolerance",
        type=float,
        default=0.18,
        help="Relative tolerance for automatic bubble-size cluster selection. Default: 0.18",
    )
    parser.add_argument(
        "--duplicate-center-factor",
        type=float,
        default=0.35,
        help="Deduplicate contours whose centers are closer than factor*median_d. Default: 0.35",
    )

    parser.add_argument(
        "--x-cluster-factor", type=float, default=0.75, help="X grouping factor. Default: 0.75"
    )
    parser.add_argument(
        "--y-cluster-factor", type=float, default=0.70, help="Y grouping factor. Default: 0.70"
    )
    parser.add_argument(
        "--column-gap-factor", type=float, default=1.80, help="Column gap factor. Default: 1.80"
    )
    parser.add_argument(
        "--column-gap-min-diam-factor",
        type=float,
        default=2.00,
        help="Minimum column gap relative to diameter. Default: 2.00",
    )
    parser.add_argument(
        "--inset-ratio", type=float, default=0.18, help="Bubble fill inset. Default: 0.18"
    )
    parser.add_argument("--debug", action="store_true", help="Write debug images and JSON reports.")
    parser.add_argument(
        "--debug-dir", default=None, help="Debug output directory. Default: ./debug_out"
    )
    return parser


def args_to_config(args: argparse.Namespace) -> Config:
    return normalize_config(
        Config(
            dpi=int(args.dpi),
            expected_options=int(args.expected_options),
            blur_ksize=int(args.blur_ksize),
            adaptive_block_size=int(args.block_size),
            adaptive_c=int(args.C),
            morph_open_ksize=int(args.morph_open_ksize),
            morph_close_ksize=int(args.morph_close_ksize),
            morph_open_iters=int(args.morph_open_iters),
            morph_close_iters=int(args.morph_close_iters),
            min_area=int(args.min_area),
            max_area=int(args.max_area),
            aspect_min=float(args.aspect_min),
            aspect_max=float(args.aspect_max),
            extent_min=float(args.extent_min),
            circularity_min=float(args.circularity_min),
            auto_size_filter=not bool(args.no_auto_size_filter),
            size_cluster_tolerance=float(args.size_cluster_tolerance),
            duplicate_center_factor=float(args.duplicate_center_factor),
            x_cluster_thresh_factor=float(args.x_cluster_factor),
            y_cluster_thresh_factor=float(args.y_cluster_factor),
            column_gap_factor=float(args.column_gap_factor),
            column_gap_min_diam_factor=float(args.column_gap_min_diam_factor),
            inset_ratio=float(args.inset_ratio),
            debug=bool(args.debug),
            debug_dir=str(args.debug_dir) if args.debug_dir else None,
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    cfg = args_to_config(args)
    setup_logger(cfg.debug)

    input_path = str(args.input)
    output_path = str(args.output)

    if not os.path.exists(input_path):
        logging.error("Input not found: %s", input_path)
        return 2

    if cfg.debug:
        cfg.debug_dir = cfg.debug_dir or "debug_out"
        safe_makedirs(cfg.debug_dir)

    try:
        answer_key = load_answer_key(args.answer_key, expected_options=cfg.expected_options)
        if is_pdf(input_path):
            if not output_path.lower().endswith(".pdf"):
                logging.warning("PDF input works best with PDF output.")
            process_pdf(input_path, output_path, cfg, answer_key)
        else:
            process_image(input_path, output_path, cfg, answer_key)
    except Exception as exc:
        logging.exception("Failed: %s", exc)
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
