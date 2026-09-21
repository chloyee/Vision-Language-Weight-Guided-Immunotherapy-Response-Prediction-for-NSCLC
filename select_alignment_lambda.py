"""Select the fixed VGMS alignment lambda using validation AUC only."""

import argparse
import json
import os
import subprocess
import sys

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_script", default="./train_vgms.py")
    parser.add_argument("--project_name", default="cohort1")
    parser.add_argument("--external_project_name", default="cohort2")
    parser.add_argument("--base_path", default="./data/patches")
    parser.add_argument("--train_split_csv", default="")
    parser.add_argument("--test_split_csv", default="")
    parser.add_argument("--output_root", default="./results/alignment_lambda_selection")
    parser.add_argument("--cache_dir", default="./results/cache")
    parser.add_argument(
        "--lambdas",
        nargs="+",
        type=float,
        default=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_size", type=float, default=2.0 / 7.0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_patches_per_case", type=int, default=1024)
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def base_command(args, results_dir, alignment_lambda, skip_test):
    command = [
        sys.executable,
        args.train_script,
        "--project_name", args.project_name,
        "--external_project_name", args.external_project_name,
        "--base_path", args.base_path,
        "--results_dir", results_dir,
        "--cache_dir", args.cache_dir,
        "--device", args.device,
        "--epochs", str(args.epochs),
        "--patience", str(args.patience),
        "--val_size", str(args.val_size),
        "--num_workers", str(args.num_workers),
        "--max_patches_per_case", str(args.max_patches_per_case),
        "--alignment_lambda", str(alignment_lambda),
        "--input_mode", "multistain",
        "--alignment_mode", "both",
        "--morphology_mode", "macenko",
        "--loss", "focal",
        "--prob_postprocess", "none",
        "--amp",
        "--tf32",
        "--no_save_train_intermediates",
    ]
    if args.train_split_csv:
        command.extend(["--train_split_csv", args.train_split_csv])
    if args.test_split_csv:
        command.extend(["--test_split_csv", args.test_split_csv])
    if skip_test:
        command.append("--skip_test_evaluation")
    return command


def read_validation_auc(result_dir):
    path = os.path.join(result_dir, "val_best_wsi_metrics.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing validation metrics: {path}")
    frame = pd.read_csv(path)
    if frame.empty or "AUC" not in frame.columns:
        raise ValueError(f"Validation metrics contain no AUC: {path}")
    return float(frame.iloc[0]["AUC"])


def main():
    args = parse_args()
    if not args.lambdas:
        raise ValueError("At least one lambda candidate is required.")
    for value in args.lambdas:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Lambda must be in [0, 1], got {value}")

    os.makedirs(args.output_root, exist_ok=True)
    rows = []
    for value in args.lambdas:
        result_dir = os.path.join(args.output_root, f"lambda_{value:.4f}")
        command = base_command(args, result_dir, value, skip_test=True)
        print(" ".join(command))
        if args.dry_run:
            continue
        subprocess.run(command, check=True)
        rows.append(
            {
                "alignment_lambda": value,
                "feature_weight": 1.0 - value,
                "morphology_weight": value,
                "validation_auc": read_validation_auc(result_dir),
                "result_dir": result_dir,
            }
        )

    if args.dry_run:
        return

    selection = pd.DataFrame(rows).sort_values(
        ["validation_auc", "alignment_lambda"],
        ascending=[False, True],
    )
    selection_path = os.path.join(args.output_root, "alignment_lambda_grid.csv")
    selection.to_csv(selection_path, index=False)
    best_lambda = float(selection.iloc[0]["alignment_lambda"])
    with open(os.path.join(args.output_root, "selected_alignment_lambda.json"), "w") as stream:
        json.dump(
            {
                "selected_alignment_lambda": best_lambda,
                "selection_metric": "validation_auc",
                "test_cohorts_used_for_selection": False,
            },
            stream,
            indent=2,
        )

    final_dir = os.path.join(args.output_root, "final_selected_model")
    final_command = base_command(args, final_dir, best_lambda, skip_test=False)
    print(f"Selected alignment_lambda={best_lambda:.4f} using validation AUC only.")
    print(" ".join(final_command))
    subprocess.run(final_command, check=True)


if __name__ == "__main__":
    main()
