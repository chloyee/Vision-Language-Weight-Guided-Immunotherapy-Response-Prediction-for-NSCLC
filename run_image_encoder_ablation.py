"""Run VGMS with pre-extracted patch features from multiple image encoders."""

import argparse
import os
import subprocess
import sys

from summary_utils import write_abtest_summary


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_script", default="./train_vgms.py")
    parser.add_argument("--project_name", default="cohort1")
    parser.add_argument("--external_project_name", default="cohort2")
    parser.add_argument("--base_path", default="./data/patches")
    parser.add_argument("--output_root", default="./results/image_encoder_ablation")
    parser.add_argument("--cache_root", default="./results/image_encoder_cache")
    parser.add_argument("--encoders", nargs="*", default=["Virchow2", "UNI-2h", "CHIEF", "TITAN"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--val_size", type=float, default=2.0 / 7.0)
    parser.add_argument("--alignment_lambda", type=float, default=0.5)
    parser.add_argument("--max_patches_per_case", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_root, exist_ok=True)
    for encoder in args.encoders:
        result_dir = os.path.join(args.output_root, encoder)
        command = [
            sys.executable, args.train_script,
            "--project_name", args.project_name,
            "--external_project_name", args.external_project_name,
            "--base_path", args.base_path,
            "--results_dir", result_dir,
            "--cache_dir", os.path.join(args.cache_root, encoder),
            "--feature_source", "wsi_pt",
            "--image_encoder_name", encoder,
            "--device", args.device,
            "--epochs", str(args.epochs),
            "--patience", str(args.patience),
            "--val_size", str(args.val_size),
            "--alignment_lambda", str(args.alignment_lambda),
            "--max_patches_per_case", str(args.max_patches_per_case),
            "--num_workers", str(args.num_workers),
            "--input_mode", "multistain",
            "--alignment_mode", "both",
            "--prob_postprocess", "none",
            "--amp", "--tf32", "--no_save_train_intermediates",
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    if not args.dry_run:
        write_abtest_summary(args.output_root, args.encoders, prefix="image_encoder_ablation_summary")


if __name__ == "__main__":
    main()
