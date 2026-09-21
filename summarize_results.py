import argparse
import glob
import os

from summary_utils import write_abtest_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Summarize internal/external WSI metrics into paper-copy CSV files."
    )
    parser.add_argument("--ablation_root", type=str, default="./soft_align_ablation_fast_results")
    parser.add_argument("--prefix", type=str, default="abtest_summary")
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional experiment directory names to summarize in this order.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    names = args.only
    if names is None:
        names = [
            os.path.basename(path)
            for path in sorted(glob.glob(os.path.join(args.ablation_root, "*")))
            if os.path.isdir(path)
        ]
    if not names:
        raise RuntimeError(f"No experiment directories found under {args.ablation_root}")
    write_abtest_summary(args.ablation_root, experiment_names=names, prefix=args.prefix)


if __name__ == "__main__":
    main()
