"""Run VGMS module ablations with validation-only checkpoint selection."""

import argparse
import os
import subprocess
import sys

from summary_utils import write_abtest_summary


ABLATIONS = [
    ("he_no_query", "he", "none", False),
    ("he_query", "he", "none", True),
    ("ihc_no_query", "ihc", "none", False),
    ("ihc_query", "ihc", "none", True),
    ("multistain_noalign_no_query", "multistain", "none", False),
    ("multistain_noalign_query", "multistain", "none", True),
    ("multistain_feature_align_no_query", "multistain", "feature", False),
    ("multistain_feature_align_query", "multistain", "feature", True),
    ("multistain_morph_align_no_query", "multistain", "morph", False),
    ("multistain_morph_align_query", "multistain", "morph", True),
    ("multistain_both_align_no_query", "multistain", "both", False),
]
FULL_MODEL = ("full_multistain_both_align_query", "multistain", "both", True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_script", default="./train_vgms.py")
    parser.add_argument("--project_name", default="cohort1")
    parser.add_argument("--external_project_name", default="cohort2")
    parser.add_argument("--base_path", default="./data/patches")
    parser.add_argument("--output_root", default="./results/module_ablation")
    parser.add_argument("--cache_dir", default="./results/cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_size", type=float, default=2.0 / 7.0)
    parser.add_argument("--alignment_lambda", type=float, default=0.5)
    parser.add_argument("--max_patches_per_case", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--include_full", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def build_command(args, experiment):
    name, input_mode, alignment_mode, use_query = experiment
    cmd = [
        sys.executable,
        args.train_script,
        "--project_name", args.project_name,
        "--external_project_name", args.external_project_name,
        "--base_path", args.base_path,
        "--results_dir", os.path.join(args.output_root, name),
        "--cache_dir", args.cache_dir,
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--val_size", str(args.val_size),
        "--alignment_lambda", str(args.alignment_lambda),
        "--max_patches_per_case", str(args.max_patches_per_case),
        "--num_workers", str(args.num_workers),
        "--input_mode", input_mode,
        "--alignment_mode", alignment_mode,
        "--prob_postprocess", "none",
        "--amp",
        "--tf32",
        "--no_save_train_intermediates",
    ]
    if not use_query:
        cmd.extend(["--disable_query_branch", "--disable_query_guidance", "--query_diversity_lambda", "0"])
    return cmd


def main():
    args = parse_args()
    experiments = list(ABLATIONS)
    if args.include_full:
        experiments.append(FULL_MODEL)
    os.makedirs(args.output_root, exist_ok=True)
    for index, experiment in enumerate(experiments, 1):
        command = build_command(args, experiment)
        print(f"[{index}/{len(experiments)}] {experiment[0]}")
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    if not args.dry_run:
        write_abtest_summary(
            args.output_root,
            experiment_names=[item[0] for item in experiments],
            prefix="module_ablation_summary",
        )


if __name__ == "__main__":
    main()
