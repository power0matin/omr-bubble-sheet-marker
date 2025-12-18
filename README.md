# PDF OMR Bubble Marker (Offline)

Offline OMR bubble marker for multiple-choice answer sheets (PDF or high-res images).  
Detects answer bubbles and fills only the correct option in pure red (`#FF0000`) while preserving the original layout.

## Features

-   Fully offline, deterministic (no ML)
-   Input: PDF (single/multi-page) or PNG/JPG
-   Output: print-quality PDF (preferred) or image
-   Bubble detection: OpenCV contour + shape heuristics
-   Grouping: multi-column, 4 options per question (left-to-right 1..4)
-   Debug mode to visualize detected bubbles and question indices

## Requirements

-   Python 3.10+
-   See `requirements.txt`

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

## Usage

### PDF -> PDF

```bash
python -m omr_marker input.pdf output.pdf --dpi 400
```

### Image -> PDF

```bash
python -m omr_marker input.png output.pdf --dpi 400
```

### Debug output

```bash
python -m omr_marker input.pdf output.pdf --dpi 400 --debug --debug-dir debug_out
```

## Notes / Assumptions

-   No rotation or perspective distortion assumed.
-   Sheet has multiple columns, exactly 4 bubbles per question.
-   Options ordered left-to-right: 1,2,3,4

## License

MIT (see `LICENSE`)
