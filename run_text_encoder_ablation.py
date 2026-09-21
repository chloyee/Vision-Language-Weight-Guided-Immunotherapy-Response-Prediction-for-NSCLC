"""Run the VGMS text-encoder ablation without using either test cohort for training."""

import argparse
import os
import subprocess
import sys

from summary_utils import write_abtest_summary


ENCODERS = {
    "clinicalbert": ["--text_encoder_type", "clinicalbert", "--text_model", "emilyalsentzer/Bio_ClinicalBERT"],
    "tfidf": ["--text_encoder_type", "tfidf"],
    "clip": ["--text_encoder_type", "clip", "--clip_text_model", "openai/clip-vit-base-patch32"],
    "qwen2.5-0.5b": ["--text_encoder_type", "qwen", "--qwen_text_model", "Qwen/Qwen2.5-0.5B"],
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_script", default="./train_vgms.py")
    parser.add_argument("--project_name", default="cohort1")
    parser.add_argument("--external_project_name", default="cohort2")
    parser.add_argument("--base_path", default="./data/patches")
    parser.add_argument("--output_root", default="./results/text_encoder_ablation")
    parser.add_argument("--cache_dir", default="./results/cache")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--val_size", type=float, default=2.0 / 7.0)
    parser.add_argument("--alignment_lambda", type=float, default=0.5)
    parser.add_argument("--max_patches_per_case", type=int, default=1024)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--prob_postprocess", choices=["none", "temperature", "shrink", "clip"], default="temperature")
    parser.add_argument("--prob_temperature", type=float, default=4.0)
    parser.add_argument("--encoders", nargs="*", default=list(ENCODERS))
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    unknown = set(args.encoders) - set(ENCODERS)
    if unknown:
        raise ValueError(f"Unknown encoders: {sorted(unknown)}")
    os.makedirs(args.output_root, exist_ok=True)
    for name in args.encoders:
        command = [
            sys.executable, args.train_script,
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
            "--input_mode", "multistain",
            "--alignment_mode", "both",
            "--prob_postprocess", args.prob_postprocess,
        "--prob_temperature", str(args.prob_temperature),
            "--amp", "--tf32", "--no_save_train_intermediates",
            *ENCODERS[name],
        ]
        print(" ".join(command))
        if not args.dry_run:
            subprocess.run(command, check=True)
    if not args.dry_run:
        write_abtest_summary(args.output_root, args.encoders, prefix="text_encoder_ablation_summary")


if __name__ == "__main__":
    main()
