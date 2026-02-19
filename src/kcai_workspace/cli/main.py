import argparse
import sys
import yaml

# Import from packages in the workspace
from kcai_data_sampling.augment_run import run_pipeline as run_augment_pipeline
from kcai_data_sampling.adversarial_run import run_pipeline as run_adversarial_pipeline
from kcai_data_sampling.kc_inference import main as run_inference_main


def handle_augment(args):
    """Handle the 'augment' subcommand."""
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    out_path = run_augment_pipeline(cfg)
    print(f"Augmentation completed. Output written to: {out_path}")


def handle_adversarial(args):
    """Handle the 'adversarial' subcommand."""
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    out_path = run_adversarial_pipeline(cfg)
    print(f"Adversarial generation completed. Output written to: {out_path}")


def handle_inference(args):
    """Handle the 'inference' subcommand."""
    # kc_inference.main() uses its own argparse, so we need to mock sys.argv
    # or refactor it. Since we want a quick unified CLI:
    original_argv = sys.argv
    sys.argv = [sys.argv[0]] + ["--config", args.config]
    if args.in_place:
        sys.argv.append("--in-place")

    try:
        run_inference_main()
    finally:
        sys.argv = original_argv


def main():
    parser = argparse.ArgumentParser(
        description="KCAI CLI: Unified entry point for data quality and robustness tools."
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")

    # Augment subcommand
    augment_parser = subparsers.add_parser("augment", help="Run data augmentation pipeline")
    augment_parser.add_argument("--config", required=True, help="Path to augmentation config YAML")
    augment_parser.set_defaults(func=handle_augment)

    # Adversarial subcommand
    adversarial_parser = subparsers.add_parser("adversarial", help="Run adversarial generation pipeline")
    adversarial_parser.add_argument("--config", required=True, help="Path to adversarial config YAML")
    adversarial_parser.set_defaults(func=handle_adversarial)

    # Inference subcommand
    inference_parser = subparsers.add_parser("inference", help="Run model inference on parquet datasets")
    inference_parser.add_argument("--config", required=True, help="Path to inference config YAML")
    inference_parser.add_argument("--in-place", action="store_true", help="Modify input file in-place")
    inference_parser.set_defaults(func=handle_inference)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
