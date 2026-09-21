import glob
import math
import os
import re

import pandas as pd


SUMMARY_COLUMNS = [
    "Experiment",
    "Internal Test Accuracy",
    "Internal Test F1",
    "Internal Test AUC (95% CI)",
    "External Test Accuracy",
    "External Test F1",
    "External Test AUC (95% CI)",
]


def _empty_metrics():
    return {"Accuracy": None, "F1": None, "AUC": None, "AUC_95_CI": ""}


def _read_metric_file(path):
    if not path or not os.path.exists(path):
        return _empty_metrics()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[WARNING] Failed to read metric file {path}: {exc}")
        return _empty_metrics()
    if df.empty:
        return _empty_metrics()
    row = df.iloc[0].to_dict()
    return {
        "Accuracy": row.get("Accuracy", None),
        "F1": row.get("F1", row.get("F1-Score", None)),
        "AUC": row.get("AUC", None),
        "AUC_95_CI": row.get("AUC_95_CI", ""),
    }


def _is_valid_number(value):
    try:
        value = float(value)
    except Exception:
        return False
    return not math.isnan(value)


def _format_number(value, digits):
    if not _is_valid_number(value):
        return ""
    return f"{float(value):.{digits}f}"


def _parse_ci(ci_value):
    if ci_value is None:
        return None, None
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", str(ci_value))
    if len(nums) < 2:
        return None, None
    return float(nums[0]), float(nums[1])


def _format_auc_ci(metrics, digits):
    auc = metrics.get("AUC", None)
    ci_low, ci_high = _parse_ci(metrics.get("AUC_95_CI", ""))
    auc_str = _format_number(auc, digits)
    if not auc_str:
        return ""
    if ci_low is None or ci_high is None:
        return auc_str
    return f"{auc_str} ({ci_low:.{digits}f}-{ci_high:.{digits}f})"


def _find_internal_metric_path(experiment_dir):
    candidates = [
        os.path.join(experiment_dir, "internal_test_best_wsi_metrics.csv"),
        os.path.join(experiment_dir, "internal_test_wsi_metrics.csv"),
        os.path.join(experiment_dir, "test_wsi_metrics.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    matches = sorted(glob.glob(os.path.join(experiment_dir, "*internal*test*best*_wsi_metrics.csv")))
    if matches:
        return matches[0]
    matches = sorted(glob.glob(os.path.join(experiment_dir, "*internal*test*_wsi_metrics.csv")))
    matches = [path for path in matches if "epoch_tmp" not in os.path.basename(path)]
    return matches[0] if matches else ""


def _find_external_metric_path(experiment_dir):
    matches = sorted(glob.glob(os.path.join(experiment_dir, "external*_wsi_metrics.csv")))
    matches = [path for path in matches if "epoch_tmp" not in os.path.basename(path)]
    return matches[0] if matches else ""


def make_summary_row(experiment_name, experiment_dir, digits):
    internal_metrics = _read_metric_file(_find_internal_metric_path(experiment_dir))
    external_metrics = _read_metric_file(_find_external_metric_path(experiment_dir))
    return {
        "Experiment": experiment_name,
        "Internal Test Accuracy": _format_number(internal_metrics.get("Accuracy"), digits),
        "Internal Test F1": _format_number(internal_metrics.get("F1"), digits),
        "Internal Test AUC (95% CI)": _format_auc_ci(internal_metrics, digits),
        "External Test Accuracy": _format_number(external_metrics.get("Accuracy"), digits),
        "External Test F1": _format_number(external_metrics.get("F1"), digits),
        "External Test AUC (95% CI)": _format_auc_ci(external_metrics, digits),
    }


def _write_rows(rows, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows, columns=SUMMARY_COLUMNS).to_csv(out_path, index=False)
    print(f"[DONE] Saved summary: {out_path}")


def write_single_experiment_summary(results_dir, experiment_name=None, prefix="summary"):
    if experiment_name is None:
        experiment_name = os.path.basename(os.path.abspath(results_dir)) or "experiment"
    rows_2dp = [make_summary_row(experiment_name, results_dir, 2)]
    rows_4dp = [make_summary_row(experiment_name, results_dir, 4)]
    path_2dp = os.path.join(results_dir, f"{prefix}_2dp.csv")
    path_4dp = os.path.join(results_dir, f"{prefix}_4dp.csv")
    _write_rows(rows_2dp, path_2dp)
    _write_rows(rows_4dp, path_4dp)
    return path_2dp, path_4dp


def write_abtest_summary(output_root, experiment_names=None, prefix="summary"):
    if experiment_names is None:
        experiment_names = [
            os.path.basename(path)
            for path in sorted(glob.glob(os.path.join(output_root, "*")))
            if os.path.isdir(path)
        ]

    rows_2dp = []
    rows_4dp = []
    for name in experiment_names:
        experiment_dir = os.path.join(output_root, name)
        if not os.path.isdir(experiment_dir):
            print(f"[WARNING] Missing experiment dir, summary row will be empty: {experiment_dir}")
        rows_2dp.append(make_summary_row(name, experiment_dir, 2))
        rows_4dp.append(make_summary_row(name, experiment_dir, 4))

    path_2dp = os.path.join(output_root, f"{prefix}_2dp.csv")
    path_4dp = os.path.join(output_root, f"{prefix}_4dp.csv")
    _write_rows(rows_2dp, path_2dp)
    _write_rows(rows_4dp, path_4dp)
    return path_2dp, path_4dp
