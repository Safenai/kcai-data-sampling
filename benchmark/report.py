"""Report assembly: CSV/JSON -> report.md."""

from __future__ import annotations

from pathlib import Path


def render_report(run_dir: Path, mode: str = "timing") -> str:
    """Render a report from the run's output files.

    Args:
        run_dir: The run's results directory.
        mode: 'timing' or 'all'.

    Returns:
        The report text.
    """
    lines = ["# Benchmark Report", f"Run: {run_dir.name}", ""]
    if (run_dir / "b1_timing.csv").exists():
        lines.append("## B1 Timing")
        lines.append((run_dir / "b1_timing.csv").read_text())
        lines.append("")
    if mode == "all" and (run_dir / "b1_memory.csv").exists():
        lines.append("## B1 Memory")
        lines.append((run_dir / "b1_memory.csv").read_text())
        lines.append("")
    return "\n".join(lines)


def write_report(run_dir: Path, mode: str = "timing") -> None:
    """Write report.md to the run directory.

    Args:
        run_dir: The run's results directory.
        mode: 'timing' or 'all'.
    """
    (run_dir / "report.md").write_text(render_report(run_dir, mode))
