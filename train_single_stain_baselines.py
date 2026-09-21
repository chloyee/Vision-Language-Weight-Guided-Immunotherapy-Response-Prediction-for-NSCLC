"""
Single-stain CHIEF MIL baselines for fixed internal train/test splits.

This script trains two independent WSI-level classifiers:
    1. HE-only CHIEF patch features
    2. IHC-only CHIEF patch features

It uses the fixed train_split.csv and test_split.csv produced by the comparison
experiments, and saves test-set probabilities for model-performance statistics.

Expected CHIEF feature layout:
    {base_path}/{project_name}/FusionFeatures_patch_level_tiles/{pair_case_name}/{tile}.pt

Each tile pt is expected to contain paired CHIEF features as [2, 768] or [1536]:
    row 0 / first 768 dims  -> HE feature
    row 1 / second 768 dims -> IHC feature
"""

import argparse
import csv
import json
import os
import random
import re
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from summary_utils import write_abtest_summary


warnings.filterwarnings("ignore")


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "case"


def strip_slide_ext(value):
    value = os.path.basename(str(value))
    return re.sub(r"\.(svs|tif|tiff|kfb|jpeg|jpg|png)$", "", value, flags=re.IGNORECASE)


def normalize_name(value):
    return strip_slide_ext(value).strip().lower()


def parse_pair_case_name(pair_case_name):
    parts = str(pair_case_name).split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return pair_case_name, pair_case_name


def normalize_pair_key(he_file, ihc_file):
    return f"{normalize_name(he_file)}__{normalize_name(ihc_file)}"


def get_he_project_name(project_name):
    project_name = str(project_name)
    return project_name[:-4] if project_name.endswith("_IHC") else project_name


def get_ihc_project_name(project_name, base_path):
    project_name = str(project_name)
    if project_name.endswith("_IHC"):
        return project_name
    candidate = project_name + "_IHC"
    if os.path.isdir(os.path.join(base_path, candidate)):
        return candidate
    return project_name


def project_candidates(project_name, base_path):
    candidates = [
        str(project_name),
        get_he_project_name(project_name),
        get_ihc_project_name(project_name, base_path),
    ]
    out = []
    for item in candidates:
        if item and item not in out:
            out.append(item)
    return out


def find_metadata_path(project_name, base_path):
    candidates = []
    for proj in project_candidates(project_name, base_path):
        candidates.extend(
            [
                os.path.join(base_path, proj, "FusionFeatures_patch_level_tiles", "pair_labels_with_case_id.csv"),
                os.path.join(base_path, proj, "pair_labels_with_case_id.csv"),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Missing metadata. Tried:\n" + "\n".join(candidates))


def find_split_csv(project_name, base_path, file_name, explicit_path=""):
    if explicit_path:
        if not os.path.exists(explicit_path):
            raise FileNotFoundError(f"Split CSV does not exist: {explicit_path}")
        return explicit_path

    candidates = []
    for proj in project_candidates(project_name, base_path):
        candidates.extend(
            [
                os.path.join(base_path, proj, "FusionFeatures_patch_level_tiles", file_name),
                os.path.join(base_path, proj, file_name),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def get_feature_dir(base_path, project_name, pair_case_name):
    candidates = []
    for proj in project_candidates(project_name, base_path):
        candidates.append(os.path.join(base_path, proj, "FusionFeatures_patch_level_tiles", pair_case_name))
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def torch_load_any(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        return obj.float()
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj).float()
    if isinstance(obj, dict):
        for key in ["features", "feature", "feat", "tensor"]:
            if key in obj:
                value = obj[key]
                if isinstance(value, torch.Tensor):
                    return value.float()
                return torch.from_numpy(np.asarray(value)).float()
    return torch.as_tensor(obj).float()


def split_he_ihc_feature(feature_tensor, feature_dim=768):
    feature_tensor = feature_tensor.float().cpu()
    if feature_tensor.ndim == 2 and feature_tensor.shape[0] >= 2:
        return feature_tensor[0, :feature_dim], feature_tensor[1, :feature_dim]
    flat = feature_tensor.reshape(-1)
    if flat.numel() < feature_dim * 2:
        raise ValueError(f"Feature tensor has too few values: {tuple(feature_tensor.shape)}")
    return flat[:feature_dim], flat[feature_dim: feature_dim * 2]


def enrich_metadata_columns(df, project_name):
    df = df.copy()
    df["project_name"] = project_name
    if "case_id" not in df.columns:
        df["case_id"] = df["pair_case_name"]
    if "he_file" not in df.columns:
        df["he_file"] = df["pair_case_name"].apply(lambda x: parse_pair_case_name(x)[0])
    if "ihc_file" not in df.columns:
        df["ihc_file"] = df["pair_case_name"].apply(lambda x: parse_pair_case_name(x)[1])
    df["label"] = df["label"].astype(int)
    df["pair_key"] = df.apply(lambda r: normalize_pair_key(r["he_file"], r["ihc_file"]), axis=1)
    df["pair_key_rev"] = df.apply(lambda r: normalize_pair_key(r["ihc_file"], r["he_file"]), axis=1)
    df["case_pair_key"] = df["case_id"].astype(str).str.lower() + "::" + df["pair_key"].astype(str)
    return df


def deduplicate_metadata(df):
    dedupe_cols = [col for col in ["case_id", "ihc_file", "he_file"] if col in df.columns]
    if len(dedupe_cols) == 3:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    df = df.drop_duplicates(subset=["pair_case_name"], keep="first").reset_index(drop=True)
    return df


def load_case_metadata(args, project_name=None, expected_cases=0):
    project_name = project_name or args.project_name
    metadata_path = find_metadata_path(project_name, args.base_path)
    print(f"[INFO] Metadata: {metadata_path}")
    df = pd.read_csv(metadata_path)
    required = {"pair_case_name", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    df = enrich_metadata_columns(df, project_name)
    df = deduplicate_metadata(df)

    ok_rows = []
    missing_examples = []
    for _, row in df.iterrows():
        feature_dir = get_feature_dir(args.base_path, project_name, str(row["pair_case_name"]))
        if os.path.isdir(feature_dir):
            ok_rows.append(row)
        elif len(missing_examples) < 10:
            missing_examples.append(feature_dir)

    if missing_examples:
        print("[WARNING] Missing feature dirs, up to 10:")
        for path in missing_examples:
            print(f"  - {path}")
    if not ok_rows:
        raise RuntimeError("No cases with existing CHIEF feature directories.")

    out = pd.DataFrame(ok_rows).reset_index(drop=True)
    print(f"[INFO] Loaded {len(out)} cases with CHIEF features from {project_name}.")
    if expected_cases > 0 and len(out) != expected_cases:
        print(
            f"[WARNING] {project_name} expected {expected_cases} cases after dedupe, "
            f"but loaded {len(out)} cases with existing CHIEF features."
        )
    return out


def normalize_split_df(split_csv):
    df = pd.read_csv(split_csv)
    df.columns = df.columns.str.strip()
    rename = {}
    for col in df.columns:
        low = col.strip().lower()
        if low in ["case id", "case_id", "case"]:
            rename[col] = "case_id"
        elif low in ["ihc file", "ihc_file", "ihc"]:
            rename[col] = "ihc_file"
        elif low in ["he file", "he_file", "he"]:
            rename[col] = "he_file"
        elif low in ["true label", "true_label", "label"]:
            rename[col] = "label"
    df = df.rename(columns=rename)
    required = {"case_id", "ihc_file", "he_file", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Split CSV {split_csv} missing columns: {missing}")

    df["case_id"] = df["case_id"].astype(str)
    df["he_file"] = df["he_file"].astype(str)
    df["ihc_file"] = df["ihc_file"].astype(str)
    df["label"] = df["label"].astype(int)
    df["split_pair_key"] = df.apply(lambda r: normalize_pair_key(r["he_file"], r["ihc_file"]), axis=1)
    df["split_pair_key_rev"] = df.apply(lambda r: normalize_pair_key(r["ihc_file"], r["he_file"]), axis=1)
    df["split_case_pair_key"] = df["case_id"].astype(str).str.lower() + "::" + df["split_pair_key"].astype(str)
    return df


def match_fixed_split(split_csv, metadata_df, results_dir, split_name):
    split_df = normalize_split_df(split_csv)
    rows = []
    unmatched = []

    by_case_pair = {str(row["case_pair_key"]): row for _, row in metadata_df.iterrows()}
    by_pair = {}
    for _, row in metadata_df.iterrows():
        by_pair.setdefault(str(row["pair_key"]), row)
        by_pair.setdefault(str(row["pair_key_rev"]), row)

    for _, split_row in split_df.iterrows():
        row = None
        key = str(split_row["split_case_pair_key"])
        if key in by_case_pair:
            row = by_case_pair[key].copy()
        else:
            key1 = str(split_row["split_pair_key"])
            key2 = str(split_row["split_pair_key_rev"])
            if key1 in by_pair:
                row = by_pair[key1].copy()
            elif key2 in by_pair:
                row = by_pair[key2].copy()

        if row is None:
            unmatched.append(split_row.to_dict())
            continue

        row["case_id"] = split_row["case_id"]
        row["he_file"] = split_row["he_file"]
        row["ihc_file"] = split_row["ihc_file"]
        row["label"] = int(split_row["label"])
        row["fixed_split"] = split_name
        rows.append(row)

    out = pd.DataFrame(rows).drop_duplicates(subset=["pair_case_name"]).reset_index(drop=True)
    if unmatched:
        path = os.path.join(results_dir, "splits", f"{split_name}_unmatched_rows.csv")
        ensure_dir(os.path.dirname(path))
        pd.DataFrame(unmatched).to_csv(path, index=False)
        print(f"[WARNING] {split_name}: {len(unmatched)} unmatched rows saved to {path}")

    print(f"[INFO] Matched {split_name}: {len(out)}/{len(split_df)} rows")
    return out


def maybe_stratify(df):
    labels = df["label"].astype(int)
    if labels.nunique() == 2 and labels.value_counts().min() >= 2:
        return labels
    return None


def split_train_val(train_df, val_size, seed):
    if val_size <= 0 or len(train_df) < 4:
        return train_df.reset_index(drop=True), train_df.iloc[0:0].copy()
    train_sub, val_df = train_test_split(
        train_df,
        test_size=val_size,
        random_state=seed,
        stratify=maybe_stratify(train_df),
    )
    return train_sub.reset_index(drop=True), val_df.reset_index(drop=True)


def build_fixed_dataframes(args):
    metadata_df = load_case_metadata(args, args.project_name)

    train_split_csv = find_split_csv(args.project_name, args.base_path, "train_split.csv", args.train_split_csv)
    test_split_csv = find_split_csv(args.project_name, args.base_path, "test_split.csv", args.test_split_csv)
    if not train_split_csv or not test_split_csv:
        raise FileNotFoundError(
            "Fixed train_split.csv/test_split.csv were not found. "
            "Please put them beside pair_labels_with_case_id.csv or pass "
            "--train_split_csv and --test_split_csv explicitly."
        )

    print(f"[INFO] Fixed train split: {train_split_csv}")
    print(f"[INFO] Fixed test split : {test_split_csv}")
    train_df = match_fixed_split(train_split_csv, metadata_df, args.results_dir, "internal_train_fixed")
    test_df = match_fixed_split(test_split_csv, metadata_df, args.results_dir, "internal_test_fixed")
    train_df, val_df = split_train_val(train_df, args.val_size, args.seed)

    split_dir = ensure_dir(os.path.join(args.results_dir, "splits"))
    train_df.to_csv(os.path.join(split_dir, "train_cases.csv"), index=False)
    val_df.to_csv(os.path.join(split_dir, "val_cases.csv"), index=False)
    test_df.to_csv(os.path.join(split_dir, "internal_test_cases.csv"), index=False)

    external_df = pd.DataFrame()
    if args.external_project_name:
        external_df = load_case_metadata(
            args,
            args.external_project_name,
            expected_cases=args.expected_external_cases,
        )
        external_df.to_csv(
            os.path.join(split_dir, f"external_{args.external_project_name}_cases.csv"),
            index=False,
        )

    print(
        f"[INFO] Split cases: train={len(train_df)}, val={len(val_df)}, "
        f"internal_test={len(test_df)}, external={len(external_df)}"
    )
    return train_df, val_df, test_df, external_df


def build_or_load_case_features(row, args, stain):
    pair_case_name = str(row["pair_case_name"])
    project_name = str(row.get("project_name", args.project_name))
    cache_path = os.path.join(
        args.cache_dir,
        stain,
        project_name,
        safe_name(pair_case_name) + ".pt",
    )
    if os.path.exists(cache_path) and not args.force_rebuild_cache:
        return torch.load(cache_path, map_location="cpu")

    feature_dir = get_feature_dir(args.base_path, project_name, pair_case_name)
    feature_paths = sorted(
        [
            os.path.join(feature_dir, name)
            for name in os.listdir(feature_dir)
            if name.lower().endswith(".pt")
        ]
    )
    if not feature_paths:
        raise RuntimeError(f"No tile feature pt files found: {feature_dir}")

    features = []
    tile_names = []
    for feature_path in feature_paths:
        tile_name = os.path.splitext(os.path.basename(feature_path))[0]
        try:
            he_feature, ihc_feature = split_he_ihc_feature(torch_load_any(feature_path), args.feature_dim)
            feature = he_feature if stain == "he" else ihc_feature
            features.append(feature)
            tile_names.append(tile_name)
        except Exception as exc:
            print(f"[WARNING] Skip {feature_path}: {exc}")

    if not features:
        raise RuntimeError(f"No usable {stain.upper()} CHIEF features for {pair_case_name}")

    he_name, ihc_name = parse_pair_case_name(pair_case_name)
    case_obj = {
        "project_name": project_name,
        "case_id": str(row.get("case_id", pair_case_name)),
        "pair_case_name": pair_case_name,
        "he_file": str(row.get("he_file", he_name)),
        "ihc_file": str(row.get("ihc_file", ihc_name)),
        "label": int(row["label"]),
        "stain": stain,
        "tile_names": tile_names,
        "features": torch.stack(features, dim=0).float(),
    }
    ensure_dir(os.path.dirname(cache_path))
    torch.save(case_obj, cache_path)
    return case_obj


class SingleStainMILDataset(Dataset):
    def __init__(self, metadata_df, args, stain, training=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.args = args
        self.stain = stain
        self.training = training

    def __len__(self):
        return len(self.metadata)

    def _sample_indices(self, n, idx):
        max_patches = int(self.args.max_patches_per_case)
        if max_patches <= 0 or n <= max_patches:
            return np.arange(n)
        if self.training:
            return np.random.choice(n, size=max_patches, replace=False)
        rng = np.random.RandomState(self.args.seed + idx)
        return np.sort(rng.choice(n, size=max_patches, replace=False))

    def __getitem__(self, idx):
        row = self.metadata.iloc[idx]
        case_obj = build_or_load_case_features(row, self.args, self.stain)
        n = int(case_obj["features"].shape[0])
        keep = self._sample_indices(n, idx)
        keep_t = torch.tensor(keep, dtype=torch.long)
        return {
            "project_name": case_obj["project_name"],
            "case_id": case_obj["case_id"],
            "pair_case_name": case_obj["pair_case_name"],
            "he_file": case_obj["he_file"],
            "ihc_file": case_obj["ihc_file"],
            "label": torch.tensor(float(case_obj["label"]), dtype=torch.float32),
            "tile_names": [case_obj["tile_names"][i] for i in keep],
            "features": case_obj["features"][keep_t],
        }


def collate_case(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 for variable-length case-level MIL.")
    return batch[0]


def make_loader(df, args, stain, training=False, shuffle=False):
    dataset = SingleStainMILDataset(df, args, stain, training=training)
    dl_args = {
        "batch_size": 1,
        "num_workers": args.num_workers,
        "collate_fn": collate_case,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if args.num_workers > 0:
        dl_args["persistent_workers"] = True
    return DataLoader(dataset, shuffle=shuffle, **dl_args)


def to_device(sample, device):
    out = dict(sample)
    out["features"] = out["features"].to(device)
    out["label"] = out["label"].to(device)
    return out


class GatedAttentionMIL(nn.Module):
    def __init__(self, in_dim=768, hidden_dim=256, attn_dim=128, dropout=0.35):
        super().__init__()
        self.patch_encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.attn_v = nn.Linear(hidden_dim, attn_dim)
        self.attn_u = nn.Linear(hidden_dim, attn_dim)
        self.attn_w = nn.Linear(attn_dim, 1)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features):
        h = self.patch_encoder(features)
        attn_logits = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(1)
        attn_weights = torch.softmax(attn_logits, dim=0)
        bag_feature = torch.sum(attn_weights.unsqueeze(1) * h, dim=0)
        logit = self.classifier(bag_feature.unsqueeze(0)).squeeze(0).squeeze(0)
        return logit, attn_weights, bag_feature


def bootstrap_auc(labels, probs, n_bootstraps=1000, seed=42):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    if len(np.unique(labels)) < 2:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.RandomState(seed)
    aucs = []
    for _ in range(n_bootstraps):
        idx = resample(np.arange(len(labels)), replace=True, random_state=rng)
        if len(np.unique(labels[idx])) < 2:
            continue
        aucs.append(roc_auc_score(labels[idx], probs[idx]))
    if not aucs:
        return float("nan"), (float("nan"), float("nan"))
    ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(labels, probs)), (float(ci_low), float(ci_high))


def compute_metrics(labels, probs, n_bootstraps=1000, seed=42):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    auc_value, auc_ci = bootstrap_auc(labels, probs, n_bootstraps=n_bootstraps, seed=seed)
    return {
        "Accuracy": accuracy_score(labels, preds),
        "AUC": auc_value,
        "AUC_95_CI": f"({auc_ci[0]:.4f}, {auc_ci[1]:.4f})",
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1": f1_score(labels, preds, zero_division=0),
        "Confusion_Matrix": str(confusion_matrix(labels, preds).tolist()),
    }


def compute_pos_weight(train_df, device):
    labels = train_df["label"].astype(int).values
    pos = max(int(labels.sum()), 1)
    neg = max(int(len(labels) - labels.sum()), 1)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def save_predictions_and_metrics(out_dir, split_name, rows, args):
    pred_df = pd.DataFrame(rows)
    pred_df["predicted_label"] = (pred_df["predict_probability"] >= 0.5).astype(int)
    preferred = [
        "project_name",
        "case_id",
        "ihc_file",
        "he_file",
        "pair_case_name",
        "true_label",
        "predicted_label",
        "predict_probability",
    ]
    pred_df = pred_df[[col for col in preferred if col in pred_df.columns]]
    pred_path = os.path.join(out_dir, f"{split_name}_wsi_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(
        labels=pred_df["true_label"].values,
        probs=pred_df["predict_probability"].values,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
    )
    pd.DataFrame([metrics]).to_csv(os.path.join(out_dir, f"{split_name}_wsi_metrics.csv"), index=False)

    print(f"\n--- {split_name} metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


def run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None):
    train = optimizer is not None
    model.train(train)
    rows = []
    total_loss = 0.0
    amp_enabled = bool(args.amp and device.type == "cuda")
    iterator = tqdm(loader, desc="Train" if train else "Eval")

    for sample in iterator:
        if sample is None:
            continue
        sample = to_device(sample, device)
        label = sample["label"].view(1)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                logit, _, _ = model(sample["features"])
                loss = criterion(logit.view(1), label)

            if train:
                if scaler is not None and scaler.is_enabled():
                    scaler.scale(loss).backward()
                    if args.grad_clip > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    if args.grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

        prob = float(torch.sigmoid(logit.detach()).cpu())
        total_loss += float(loss.detach().cpu())
        rows.append(
            {
                "project_name": sample["project_name"],
                "case_id": sample["case_id"],
                "ihc_file": sample["ihc_file"],
                "he_file": sample["he_file"],
                "pair_case_name": sample["pair_case_name"],
                "true_label": float(label.detach().cpu()),
                "predict_probability": prob,
            }
        )
        iterator.set_postfix(loss=float(loss.detach().cpu()), prob=prob)

    return total_loss / max(len(loader), 1), rows


@torch.no_grad()
def evaluate_and_save(model, loader, criterion, device, args, out_dir, split_name):
    loss, rows = run_epoch(model, loader, criterion, device, args, optimizer=None, scaler=None)
    metrics = save_predictions_and_metrics(out_dir, split_name, rows, args)
    return loss, metrics, rows


def train_one_stain(stain, train_df, val_df, test_df, external_df, args, device):
    out_dir = ensure_dir(os.path.join(args.results_dir, f"{stain}_chief"))
    save_json(vars(args), os.path.join(out_dir, "config.json"))

    train_loader = make_loader(train_df, args, stain, training=True, shuffle=True)
    val_loader = make_loader(val_df, args, stain, training=False, shuffle=False)
    test_loader = make_loader(test_df, args, stain, training=False, shuffle=False)
    external_loader = (
        make_loader(external_df, args, stain, training=False, shuffle=False)
        if external_df is not None and len(external_df) > 0
        else None
    )
    if len(val_df) == 0:
        raise ValueError(
            "A non-empty validation split is required. Internal and external "
            "test cohorts are never used for checkpoint selection."
        )
    selection_name = "val"
    selection_loader = val_loader
    selection_df = val_df

    model = GatedAttentionMIL(
        in_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        attn_dim=args.attn_dim,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_df, device)).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=max(2, args.lr_patience),
        min_lr=args.min_lr,
        verbose=True,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(out_dir, "best_model.pt")

    for epoch in range(1, args.epochs + 1):
        print("=" * 100)
        print(f"[{stain.upper()}] Epoch {epoch}/{args.epochs}")
        train_loss, train_rows = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            args,
            optimizer=optimizer,
            scaler=scaler,
        )
        train_metrics = compute_metrics(
            labels=[row["true_label"] for row in train_rows],
            probs=[row["predict_probability"] for row in train_rows],
            n_bootstraps=100,
            seed=args.seed,
        )

        if selection_name == "train":
            selection_metrics = train_metrics
        else:
            _, selection_rows = run_epoch(model, selection_loader, criterion, device, args, optimizer=None, scaler=None)
            selection_metrics = compute_metrics(
                labels=[row["true_label"] for row in selection_rows],
                probs=[row["predict_probability"] for row in selection_rows],
                n_bootstraps=100,
                seed=args.seed,
            )

        score = selection_metrics["AUC"] if not np.isnan(selection_metrics["AUC"]) else selection_metrics["Accuracy"]
        print(
            f"[INFO] train_loss={train_loss:.4f}, train_auc={train_metrics['AUC']}, "
            f"selection={selection_name}, selection_auc={selection_metrics['AUC']}, "
            f"selection_acc={selection_metrics['Accuracy']:.4f}, selection_n={len(selection_df)}"
        )

        scheduler.step(score)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save({"model": model.state_dict(), "args": vars(args), "stain": stain}, best_path)
            print(f"[INFO] New best score={best_score:.4f}; saved {best_path}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. patience={patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    print(f"[INFO] Loading best {stain.upper()} model from epoch {best_epoch}: {best_path}")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    evaluate_and_save(model, train_loader, criterion, device, args, out_dir, "train_best")
    if len(val_df) > 0:
        evaluate_and_save(model, val_loader, criterion, device, args, out_dir, "val_best")
    _, _, test_rows = evaluate_and_save(model, test_loader, criterion, device, args, out_dir, "internal_test")
    external_rows = []
    if external_loader is not None:
        _, _, external_rows = evaluate_and_save(
            model,
            external_loader,
            criterion,
            device,
            args,
            out_dir,
            f"external_{args.external_project_name}",
        )
    return out_dir, test_rows, external_rows


def merge_stain_probabilities(results_dir, stain_to_rows, split_name, out_file_name):
    if not {"he", "ihc"}.issubset(set(stain_to_rows.keys())):
        return

    he_df = pd.DataFrame(stain_to_rows["he"])
    ihc_df = pd.DataFrame(stain_to_rows["ihc"])
    if he_df.empty or ihc_df.empty:
        return
    keys = ["project_name", "case_id", "ihc_file", "he_file", "pair_case_name", "true_label"]
    he_cols = keys + ["predict_probability"]
    ihc_cols = keys + ["predict_probability"]
    he_df = he_df[he_cols].rename(
        columns={
            "predict_probability": "he_predict_probability",
        }
    )
    ihc_df = ihc_df[ihc_cols].rename(
        columns={
            "predict_probability": "ihc_predict_probability",
        }
    )
    merged = he_df.merge(ihc_df, on=keys, how="outer")
    merged["he_predicted_label"] = (merged["he_predict_probability"] >= 0.5).astype("Int64")
    merged["ihc_predicted_label"] = (merged["ihc_predict_probability"] >= 0.5).astype("Int64")
    out_path = os.path.join(results_dir, out_file_name)
    merged.to_csv(out_path, index=False)
    print(f"[DONE] Saved merged HE/IHC {split_name} probabilities for statistics: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train HE-only and IHC-only CHIEF attention-MIL baselines on fixed splits."
    )
    parser.add_argument("--project_name", type=str, default="cohort1")
    parser.add_argument("--external_project_name", type=str, default="cohort2")
    parser.add_argument(
        "--expected_external_cases",
        type=int,
        default=101,
        help="Warn if external project case count after dedupe differs from this. Set 0 to disable.",
    )
    parser.add_argument("--base_path", type=str, default="../data/patches")
    parser.add_argument("--results_dir", type=str, default="./single_stain_chief_baselines")
    parser.add_argument("--cache_dir", type=str, default="./single_stain_chief_baselines/case_tensor_cache")
    parser.add_argument("--train_split_csv", type=str, default="")
    parser.add_argument("--test_split_csv", type=str, default="")
    parser.add_argument("--stains", nargs="*", default=["he", "ihc"], choices=["he", "ihc"])

    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--attn_dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument("--max_patches_per_case", type=int, default=1024)

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr_patience", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--grad_clip", type=float, default=5.0)
    parser.add_argument("--val_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstraps", type=int, default=1000)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    ensure_dir(args.results_dir)
    ensure_dir(args.cache_dir)
    save_json(vars(args), os.path.join(args.results_dir, "config.json"))

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"[INFO] Device: {device}")
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print("[INFO] TF32 enabled.")

    train_df, val_df, test_df, external_df = build_fixed_dataframes(args)

    internal_stain_to_rows = {}
    external_stain_to_rows = {}
    experiment_names = []
    for stain in args.stains:
        out_dir, test_rows, external_rows = train_one_stain(
            stain,
            train_df,
            val_df,
            test_df,
            external_df,
            args,
            device,
        )
        internal_stain_to_rows[stain] = test_rows
        external_stain_to_rows[stain] = external_rows
        experiment_names.append(os.path.basename(out_dir))

    merge_stain_probabilities(
        args.results_dir,
        internal_stain_to_rows,
        split_name="internal_test",
        out_file_name="single_stain_chief_internal_test_probabilities_for_stats.csv",
    )
    internal_new = os.path.join(args.results_dir, "single_stain_chief_internal_test_probabilities_for_stats.csv")
    internal_legacy = os.path.join(args.results_dir, "single_stain_chief_test_probabilities_for_stats.csv")
    if os.path.exists(internal_new):
        shutil.copyfile(internal_new, internal_legacy)
        print(f"[DONE] Saved legacy internal-test probability filename: {internal_legacy}")
    merge_stain_probabilities(
        args.results_dir,
        external_stain_to_rows,
        split_name=f"external_{args.external_project_name}",
        out_file_name=f"single_stain_chief_external_{args.external_project_name}_probabilities_for_stats.csv",
    )
    write_abtest_summary(
        args.results_dir,
        experiment_names=experiment_names,
        prefix="single_stain_chief_summary",
    )
    print(f"[DONE] Results saved to: {args.results_dir}")


if __name__ == "__main__":
    main()
