import subprocess
import sys

def test_cli_help():
    r = subprocess.run([sys.executable, "-m", "omr_marker", "-h"], capture_output=True, text=True)
    assert r.returncode == 0
    assert "Offline OMR bubble marker" in (r.stdout + r.stderr)
