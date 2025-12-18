# OMR Bubble Sheet Marker (Offline)

[![CI](https://github.com/power0matin/omr-bubble-sheet-marker/actions/workflows/ci.yml/badge.svg)](https://github.com/power0matin/omr-bubble-sheet-marker/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Offline](https://img.shields.io/badge/Mode-Offline-important)

**Offline, deterministic** bubble-marker for standard multiple-choice answer sheets (OMR).  
Accepts **PDF (single/multi-page)** or **high-resolution images (PNG/JPG)** and produces an annotated output (preferably a **print-quality PDF**) by filling **only the correct option** bubble in **pure red** `#FF0000`.

> This project is designed to be a clean foundation for an open-source or internal OMR pipeline (no ML, no cloud).

## Table of Contents

-   [Key Features](#key-features)
-   [Assumptions & Constraints](#assumptions--constraints)
-   [Quickstart](#quickstart)
-   [Installation](#installation)
-   [Usage](#usage)
-   [CLI Options](#cli-options)
-   [Answer Key](#answer-key)
-   [Debug Mode](#debug-mode)
-   [Tuning Guide](#tuning-guide)
-   [Troubleshooting](#troubleshooting)
-   [Repository Layout](#repository-layout)
-   [Development](#development)
-   [Roadmap](#roadmap)
-   [License](#license)

## Key Features

-   **Fully offline** (no API, no cloud, no ML)
-   Inputs:
    -   **PDF** (single/multi-page) rendered via PyMuPDF at configurable **DPI**
    -   **PNG/JPG** images (recommended: 300+ DPI equivalent)
-   Output:
    -   Preferred: **PDF** with print-quality resolution and unchanged layout
    -   Alternative: **high-resolution image**
-   Deterministic OpenCV pipeline:
    -   grayscale → Gaussian blur → adaptive threshold (binary inverted) → morphology
    -   contour detection + geometric heuristics (area/aspect/circularity/extent)
-   Grouping logic:
    -   multi-column sheets
    -   **exactly 4 bubbles per question**
    -   options ordered **left-to-right**: 1,2,3,4
    -   question numbering: **top-to-bottom, column-by-column**
-   Marking:
    -   fills **only the correct bubble** with **pure red** `#FF0000`
    -   preserves bubble outlines via an **inset fill** strategy
-   Resilient behavior:
    -   rows that don’t contain exactly 4 bubbles are **skipped**
    -   unexpected layouts produce **warnings**, not crashes
-   Debug mode produces per-page visual overlays (detected bubbles + question indices)

## Assumptions & Constraints

This repository targets standard OMR-style multiple-choice sheets with:

-   Input quality: **300 DPI equivalent or higher**
-   **No rotation** and **no perspective distortion** assumed
-   Multiple columns are supported
-   Each question contains **exactly 4 bubbles** (oval/elliptical/circular)
-   Bubble order is **left-to-right** mapping to options **1..4**
-   This tool **does not detect student-filled answers** (it only marks the provided answer key)

> If your scans are slightly skewed, consider deskewing upstream. A deskew utility is listed in the [Roadmap](#roadmap).

## Quickstart

### PDF → PDF (recommended)

```bash
python -m omr_marker input.pdf output.pdf --dpi 400
```

### Image → PDF

```bash
python -m omr_marker input.png output.pdf --dpi 400
```

### Debug output

```bash
python -m omr_marker input.pdf output.pdf --dpi 400 --debug --debug-dir debug_out
```

## Installation

### Requirements

-   Python **3.10+**
-   Offline execution

### Create a virtual environment & install dependencies

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
pip install -r requirements.txt
```

#### Windows (PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

## Usage

### Run as a module (recommended)

```bash
python -m omr_marker input.pdf output.pdf --dpi 400
```

### Run via file path (if needed)

```bash
python src/omr_marker/mark_answers.py input.pdf output.pdf --dpi 400
```

### Get help

```bash
python -m omr_marker -h
```

## CLI Options

Most users can start with defaults. For non-standard scans, use the tuning knobs below.

| Option                         |     Default | Description                                       |
| ------------------------------ | ----------: | ------------------------------------------------- |
| `--dpi`                        |         400 | PDF render DPI and assumed DPI for image→PDF      |
| `--debug`                      |       False | Enable debug overlays                             |
| `--debug-dir`                  | `debug_out` | Directory for debug outputs                       |
| `--blur-ksize`                 |           5 | Gaussian blur kernel size (odd)                   |
| `--block-size`                 |          35 | Adaptive threshold block size (odd)               |
| `--C`                          |           8 | Adaptive threshold C parameter                    |
| `--morph-open-ksize`           |           3 | Morph open kernel size                            |
| `--morph-close-ksize`          |           3 | Morph close kernel size                           |
| `--morph-open-iters`           |           1 | Morph open iterations                             |
| `--morph-close-iters`          |           1 | Morph close iterations                            |
| `--min-area`                   |         300 | Minimum contour area to consider a bubble         |
| `--max-area`                   |       20000 | Maximum contour area to consider a bubble         |
| `--aspect-min`                 |        0.55 | Min bbox aspect ratio (w/h)                       |
| `--aspect-max`                 |        1.80 | Max bbox aspect ratio (w/h)                       |
| `--extent-min`                 |        0.35 | Min extent = area/(w\*h)                          |
| `--circularity-min`            |        0.35 | Min circularity (shape heuristic)                 |
| `--x-cluster-factor`           |        0.75 | X clustering threshold factor × median diameter   |
| `--y-cluster-factor`           |        0.70 | Y clustering threshold factor × median diameter   |
| `--column-gap-factor`          |        1.80 | Column split threshold factor × median X gap      |
| `--column-gap-min-diam-factor` |        2.00 | Minimum column split gap factor × median diameter |
| `--inset-ratio`                |        0.18 | Inset ratio to preserve bubble outline            |

## Answer Key

The current version ships with an **in-code** answer key for deterministic behavior and quick testing.

<details>
<summary><strong>Show answer key (30 questions)</strong></summary>

|   Q |   A |   Q |   A |   Q |   A |
| --: | --: | --: | --: | --: | --: |
|   1 |   1 |  11 |   4 |  21 |   3 |
|   2 |   4 |  12 |   2 |  22 |   4 |
|   3 |   4 |  13 |   1 |  23 |   1 |
|   4 |   1 |  14 |   3 |  24 |   3 |
|   5 |   1 |  15 |   3 |  25 |   1 |
|   6 |   3 |  16 |   4 |  26 |   4 |
|   7 |   2 |  17 |   3 |  27 |   2 |
|   8 |   1 |  18 |   3 |  28 |   4 |
|   9 |   2 |  19 |   3 |  29 |   2 |
|  10 |   3 |  20 |   4 |  30 |   3 |

</details>

> Recommended improvement: support `--answer-key answer_key.json` to avoid code changes (see [Roadmap](#roadmap)).

## Debug Mode

Enable debug overlays to validate detection & grouping:

```bash
python -m omr_marker input.pdf output.pdf --dpi 400 --debug --debug-dir debug_out
```

Typical outputs:

```text
debug_out/page_1_debug.png
debug_out/page_2_debug.png
```

Debug overlays include:

-   bounding boxes for detected bubbles
-   question row boxes
-   question indices (Q1, Q2, ...)

## Tuning Guide

<details>
<summary><strong>If too few bubbles are detected</strong></summary>

1. Increase DPI (PDF input):

```bash
python -m omr_marker input.pdf output.pdf --dpi 600
```

2. Relax geometric filters:

-   decrease `--min-area` (e.g., 300 → 200)
-   decrease `--circularity-min` slightly (e.g., 0.35 → 0.30)

3. Improve binarization:

-   increase `--block-size` (e.g., 35 → 45 or 55)
-   tweak `--C` (try 6..12)
-   increase `--morph-close-ksize` (e.g., 3 → 5) to close small gaps

</details>

<details>
<summary><strong>If too many non-bubble contours pass the filters</strong></summary>

1. Tighten geometric filters:

-   increase `--min-area`
-   increase `--circularity-min` (e.g., 0.35 → 0.45)
-   increase `--extent-min` (e.g., 0.35 → 0.45)
-   narrow aspect range (e.g., `--aspect-min 0.70 --aspect-max 1.40`)

2. Reduce noise in thresholding:

-   increase `--morph-open-iters`
-   slightly increase `--morph-open-ksize`

</details>

<details>
<summary><strong>If grouping is wrong (rows not forming sets of 4)</strong></summary>

Grouping is sensitive to scan consistency and bubble spacing.

-   If bubbles over-merge into the same row/cluster: decrease `--y-cluster-factor`
-   If rows split too aggressively: increase `--y-cluster-factor`
-   If option columns merge: decrease `--x-cluster-factor`
-   If option columns split incorrectly: increase `--x-cluster-factor`

Always validate via `--debug` first before making large changes.

</details>

## Troubleshooting

### OpenCV import errors on Linux (libGL / runtime deps)

On Ubuntu/Debian:

```bash
sudo apt-get update
sudo apt-get install -y libgl1 libglib2.0-0
```

### Output PDF is generated but nothing is marked

-   Turn on debug mode to confirm bubble detection:

    ```bash
    python -m omr_marker input.pdf output.pdf --dpi 400 --debug
    ```

-   Increase DPI to 600
-   Adjust `--min-area` / `--max-area` based on bubble size
-   Confirm your layout matches the [Assumptions & Constraints](#assumptions--constraints)

### Marking works but question order is incorrect

This project assumes question numbering:

-   **top-to-bottom**, then **column-by-column** (left column first)

If your sheet uses a different order (e.g., row-major across the full page), the numbering logic must be adapted (see [Roadmap](#roadmap)).

## Repository Layout

```text
omr-bubble-sheet-marker/
├─ src/omr_marker/
│  ├─ __init__.py
│  ├─ __main__.py
│  └─ mark_answers.py
├─ tests/
├─ .github/workflows/ci.yml
├─ requirements.txt
├─ requirements-dev.txt
├─ pyproject.toml
└─ README.md
```

## Development

### Install dev dependencies

```bash
pip install -r requirements-dev.txt
pip install -e .
```

### Lint / format / tests

```bash
ruff check .
black --check .
pytest -q
```

## Roadmap

-   [ ] Load answer key from external file: `--answer-key answer_key.json` (JSON/YAML)
-   [ ] Structured debug report (`debug.json`) with detection/grouping stats
-   [ ] Optional deskew (non-ML) for mildly rotated scans
-   [ ] Alternate numbering schemes (row-major / custom layouts)
-   [ ] Golden test fixtures (synthetic sheets) for regression testing

## 📬 Contact

**Matin Shahabadi (متین شاه‌آبادی / متین شاه آبادی)**

-   Website: [matinshahabadi.ir](https://matinshahabadi.ir)
-   Email: [me@matinshahabadi.ir](mailto:me@matinshahabadi.ir)
-   GitHub: [power0matin](https://github.com/power0matin)
-   LinkedIn: [matin-shahabadi](https://www.linkedin.com/in/matin-shahabadi)

## License

MIT — see [LICENSE](./LICENSE)

[![Stargazers over time](https://starchart.cc/power0matin/discord-quest-auto-completer.svg?variant=adaptive)](https://starchart.cc/power0matin/discord-quest-auto-completer)
