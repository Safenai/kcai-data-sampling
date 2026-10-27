"""Benchmark CLI and orchestration."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmark.orchestrate import orchestrate


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m benchmark", description="Albumentations A vs D benchmark")
    p.add_argument("--quick", action="store_true", help="1 warmup + 3 timed (dev mode)")
    p.add_argument("--only", choices=["s1", "s2", "s3"], help="Run only one suite")
    p.add_argument("--repeats", type=int, help="Override timed repeats")
    p.add_argument("--warmups", type=int, help="Override warmups")
    p.add_argument("--with-attribution", action="store_true", help="Run M3/M5 attribution (separate runs)")
    p.add_argument("--rebuild-inputs", action="store_true", help="Regenerate S1 inputs")
    p.add_argument("--dry-run", action="store_true", help="Print run matrix and write run_plan.json without running")
    p.add_argument("--outputs-root", type=Path, default=Path("benchmark/outputs"), help="Outputs directory")
    p.add_argument("--report", choices=["timing", "all"], default="timing", help="Report mode")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.dry_run:
        print("DRY RUN: would execute benchmark")
        print(f"  quick={args.quick}, only={args.only}, report={args.report}")
        return 0

    run_dir = orchestrate(
        outputs_root=args.outputs_root,
        quick=args.quick,
        only=args.only,
        report_mode=args.report,
        with_attribution=args.with_attribution,
    )
    print(f"Benchmark completed. Results in: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
