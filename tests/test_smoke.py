import subprocess
import sys

import cv2
import numpy as np


def test_cli_help():
    result = subprocess.run(
        [sys.executable, "-m", "omr_marker", "-h"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0
    assert "Offline OMR bubble marker" in (result.stdout + result.stderr)


def _write_synthetic_sheet(path):
    img = np.full((1650, 1200, 3), 255, dtype=np.uint8)
    cv2.putText(img, "OMR TEST", (455, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

    bubble_r = 18
    row_gap = 48
    col_specs = [(1, 110, 210), (16, 650, 750)]
    option_gap = 58
    start_y = 170

    for q_start, label_x, bubble_x in col_specs:
        for option in range(4):
            x = bubble_x + option * option_gap
            cv2.putText(
                img,
                str(option + 1),
                (x - 6, start_y - 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0),
                1,
            )
        for row in range(15):
            q = q_start + row
            y = start_y + row * row_gap
            cv2.putText(
                img, f"{q:02d}", (label_x, y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1
            )
            for option in range(4):
                x = bubble_x + option * option_gap
                cv2.circle(img, (x, y), bubble_r, (0, 0, 0), 3)

    assert cv2.imwrite(str(path), img)


def test_synthetic_sheet_marks_all_questions(tmp_path):
    input_path = tmp_path / "sheet.png"
    output_path = tmp_path / "marked.png"
    debug_dir = tmp_path / "debug"
    _write_synthetic_sheet(input_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "omr_marker",
            str(input_path),
            str(output_path),
            "--debug",
            "--debug-dir",
            str(debug_dir),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Marked 30 bubble(s)." in (result.stdout + result.stderr)
    assert output_path.exists()
    assert (debug_dir / "image_debug.png").exists()
    assert (debug_dir / "image_report.json").exists()

    out = cv2.imread(str(output_path), cv2.IMREAD_COLOR)
    assert out is not None
    red_mask = (out[:, :, 2] > 180) & (out[:, :, 1] < 80) & (out[:, :, 0] < 80)
    assert int(red_mask.sum()) > 1000
