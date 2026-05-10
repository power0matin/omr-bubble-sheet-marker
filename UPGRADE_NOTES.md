# OMR Marker Optimized Build

This build focuses on making the OMR marker more reliable on real answer-sheet images.

## Main improvements

- Uses `cv2.RETR_LIST` instead of `cv2.RETR_EXTERNAL`, so bubbles are still detected when the page has an outer border.
- Adds dominant bubble-size filtering to prevent question numbers, option labels, and title text from being treated as bubbles.
- Deduplicates inner/outer contours and keeps the outer bubble boundary.
- Fixes image-to-PDF color handling so `#FF0000` remains red instead of becoming blue.
- Adds `--answer-key answer_key.json` support.
- Adds `--expected-options` for configurable option count per question.
- Adds JSON debug reports beside debug images.
- Adds a synthetic regression test that verifies all 30 questions are marked.
- Updates project version to `0.2.0`.

## Recommended command

```powershell
python -m omr_marker ChatGPT.png ChatGPT_marked.pdf --dpi 400 --debug --debug-dir debug_out
```

After running, check:

```text
debug_out\image_debug.png
debug_out\image_report.json
ChatGPT_marked.pdf
```

## Answer-key JSON examples

Object format:

```json
{
  "1": 1,
  "2": 4,
  "3": 4
}
```

List format:

```json
[1, 4, 4, 1, 1]
```

Run with:

```powershell
python -m omr_marker ChatGPT.png output.pdf --answer-key answer_key.json
```
