from pathlib import Path
import runpy
import sys


def test_comparison_script_runs_minimal(monkeypatch):
    script = Path(__file__).resolve().parents[1] / "examples" / "compare_with_fourier_filters.py"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--coefficients",
            "5",
            "--repetitions",
            "1",
            "--grid-size",
            "256",
            "--format",
            "json",
        ],
    )
    try:
        runpy.run_path(str(script), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0
