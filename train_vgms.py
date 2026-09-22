"""
VGMS paper-protocol training for paired H&E and PD-L1 IHC WSIs.

This script expects de-identified cohort folders where CHIEF patch features
have already been extracted:

    ../data/patches/{PROJECT}/FusionFeatures_patch_level_tiles/{pair_case_name}/{tile}.pt

Each tile feature is expected to be either [2, 768] or flattened [1536]:
    row 0 / first 768 dims  -> HE feature
    row 1 / second 768 dims -> IHC feature

No clinical variables are used.

Each sampled IHC anchor softly aggregates the Top-K H&E candidates from the
same case. Model input after distance-aware fixed-lambda alignment is:
    [weighted_HE_i, IHC_i, weighted_HE_hematoxylin_i, IHC_DAB_i]

The model saves:
    - cached case tensors with CHIEF features and stain morphology descriptors
    - query embeddings from ClinicalBERT / TF-IDF / CLIP / Qwen2.5-0.5B
    - trained checkpoints
    - WSI-level predictions and metrics
    - per-case intermediate npz files: alignment scores/indices, query weights,
      query-specific evidence, and final features.
"""

import argparse
import csv
import json
import math
import os
import random
import re
import warnings
from dataclasses import asdict, dataclass

import cv2
import numpy as np
import pandas as pd
import torch
from safe_spiral import bounded_spiral_align
from stain_scaling import SCALAR_SCHEMA, prepare_stain_scaler, scale_case
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image
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

from summary_utils import write_single_experiment_summary


warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None


DEFAULT_QUERIES = [
    (
        "Q1_myeloid_inflammatory_aggregates",
        "macrophage- and neutrophil-rich myeloid inflammatory aggregates in tumor stroma",
    ),
    (
        "Q2_strong_membranous_pdl1_dab",
        "strong membranous PD-L1 DAB staining in tumor cells",
    ),
    (
        "Q3_tils_within_tumor_nests",
        "tumor-infiltrating lymphocytes within tumor nests",
    ),
    (
        "Q4_tumor_stroma_interface_lymphocytes",
        "tumor-stroma interface with lymphocytic infiltration",
    ),
    (
        "Q5_immune_cold_few_lymphocytes",
        "immune-cold tumor regions with few lymphocytes",
    ),
    (
        "Q6_fibroblast_rich_desmoplastic_stroma",
        "fibroblast-rich desmoplastic stroma surrounding invasive tumor nests",
    ),
]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def safe_name(text):
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_") or "case"


def save_json(obj, path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_cache_dir(args):
    if getattr(args, "cache_dir", ""):
        return ensure_dir(args.cache_dir)
    return ensure_dir(os.path.join(args.results_dir, "case_tensor_cache"))


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
    return flat[:feature_dim], flat[feature_dim:feature_dim * 2]


def tile_coordinate_yx(tile_name):
    """Read y_x tile names in the shared, already scale-mapped coordinate frame."""
    stem = os.path.splitext(os.path.basename(str(tile_name)))[0]
    match = re.fullmatch(r"(-?\d+)_(-?\d+)", stem)
    if match is None:
        raise ValueError(f"Expected y_x tile name in shared coordinates, got {tile_name!r}")
    return int(match.group(1)), int(match.group(2))


def spiral_candidate_order(candidate_yx, origin_yx):
    """Clockwise square spiral, starting at the origin then moving right.

    Sort only retained tissue coordinates, skipping empty grid locations.
    Increasing Chebyshev radius gives expanding square rings. Within each
    ring the order is right edge down, bottom left, left up, then top right.
    Coordinates can be grid indices or common-frame pixels without a stride.
    """
    oy, ox = origin_yx

    def key(index):
        y, x = candidate_yx[index]
        dy, dx = y - oy, x - ox
        radius = max(abs(dy), abs(dx))
        if radius == 0:
            return (0, 0, index)
        if dx == radius and dy > -radius:
            offset = dy + radius - 1
        elif dy == radius:
            offset = 2 * radius + (radius - 1 - dx)
        elif dx == -radius:
            offset = 4 * radius + (radius - 1 - dy)
        else:
            offset = 6 * radius + (dx + radius - 1)
        return (radius, offset, index)

    return sorted(range(len(candidate_yx)), key=key)


def parse_pair_case_name(pair_case_name):
    parts = str(pair_case_name).split("__")
    if len(parts) >= 2:
        return parts[0], parts[1]
    return pair_case_name, pair_case_name


def strip_slide_ext(value):
    value = os.path.basename(str(value))
    return re.sub(r"\.(svs|tif|tiff|kfb|jpeg|jpg|png)$", "", value, flags=re.IGNORECASE)


def normalize_name(value):
    return strip_slide_ext(value).strip().lower()


def normalize_pair_key(he_file, ihc_file):
    return f"{normalize_name(he_file)}__{normalize_name(ihc_file)}"


def get_he_project_name(project_name):
    project_name = str(project_name)
    if project_name.endswith("_IHC"):
        return project_name[:-4]
    return project_name


def get_ihc_project_name(project_name, base_path=None):
    project_name = str(project_name)
    if project_name.endswith("_IHC"):
        return project_name
    if base_path:
        candidate = project_name + "_IHC"
        if os.path.isdir(os.path.join(base_path, candidate)):
            return candidate
    return project_name


def get_wsi_feature_dir(project_name, encoder_name, stain, args):
    stain = str(stain).lower()
    if stain == "he":
        feature_project = get_he_project_name(project_name)
    elif stain == "ihc":
        feature_project = get_ihc_project_name(project_name, args.base_path)
    else:
        raise ValueError(f"Unknown stain: {stain}")
    return os.path.join(args.base_path, feature_project, f"{encoder_name}_features")


def get_wsi_feature_paths(pair_case_name, project_name, encoder_name, args, he_file="", ihc_file=""):
    he_case_name, ihc_case_name = parse_pair_case_name(pair_case_name)
    he_name = strip_slide_ext(he_file or he_case_name)
    ihc_name = strip_slide_ext(ihc_file or ihc_case_name)
    he_path = os.path.join(get_wsi_feature_dir(project_name, encoder_name, "he", args), f"{he_name}.pt")
    ihc_path = os.path.join(get_wsi_feature_dir(project_name, encoder_name, "ihc", args), f"{ihc_name}.pt")
    return he_path, ihc_path


def load_wsi_feature_matrix(path):
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, torch.Tensor):
        feat = obj.float()
    elif isinstance(obj, np.ndarray):
        feat = torch.from_numpy(obj).float()
    elif isinstance(obj, dict):
        feat = None
        for key in ["features", "feature", "feat", "feats", "embeddings", "embedding", "x"]:
            if key in obj:
                feat = obj[key]
                break
        if feat is None:
            raise ValueError(f"{path} is a dict but no feature key was found; keys={list(obj.keys())}")
        if isinstance(feat, torch.Tensor):
            feat = feat.float()
        elif isinstance(feat, np.ndarray):
            feat = torch.from_numpy(feat).float()
        else:
            feat = torch.as_tensor(feat).float()
    else:
        feat = torch.as_tensor(obj).float()

    if feat.ndim == 1:
        feat = feat.unsqueeze(0)
    elif feat.ndim == 3 and feat.shape[0] == 1:
        feat = feat.squeeze(0)
    elif feat.ndim >= 3:
        feat = feat.reshape(-1, feat.shape[-1])

    if feat.ndim != 2:
        raise ValueError(f"{path} cannot be converted to [N, D], got shape={tuple(feat.shape)}")
    return feat.float().cpu()


def metadata_pair_keys(pair_case_name):
    first, second = parse_pair_case_name(pair_case_name)
    return {
        normalize_pair_key(first, second),
        normalize_pair_key(second, first),
    }


def find_image_file(roots, tile_name):
    stem = os.path.splitext(os.path.basename(str(tile_name)))[0]
    candidates = []
    for root in roots:
        candidates.extend(
            [
                os.path.join(root, stem + ".png"),
                os.path.join(root, stem + ".jpg"),
                os.path.join(root, stem + ".jpeg"),
                os.path.join(root, stem + ".tif"),
                os.path.join(root, stem + ".tiff"),
            ]
        )
    for path in candidates:
        if os.path.exists(path):
            return path
    return ""


def get_patch_roots(base_path, project_name, he_case_name, ihc_case_name, patch_size=250):
    root = os.path.join(base_path, project_name)
    he_project = get_he_project_name(project_name)
    ihc_project = get_ihc_project_name(project_name, base_path)
    he_root = os.path.join(base_path, he_project)
    ihc_root = os.path.join(base_path, ihc_project)
    patch_roots = [
        os.path.join(root, f"patches_lv0_ps{patch_size}"),
        os.path.join(root, "patches_lv0_ps256"),
    ]
    he_patch_roots = [
        os.path.join(he_root, f"patches_lv0_ps{patch_size}"),
        os.path.join(he_root, "patches_lv0_ps256"),
    ]
    ihc_patch_roots = [
        os.path.join(ihc_root, f"patches_lv0_ps{patch_size}"),
        os.path.join(ihc_root, "patches_lv0_ps256"),
    ]

    he_roots = []
    for he_patch_root in he_patch_roots:
        he_roots.extend(
            [
                os.path.join(he_patch_root, f"{he_project}_slides_data", he_case_name),
                os.path.join(he_patch_root, f"{he_project}_HE_slides_data", he_case_name),
                os.path.join(he_patch_root, f"{project_name}_slides_data", he_case_name),
                os.path.join(he_patch_root, f"{project_name}_HE_slides_data", he_case_name),
                os.path.join(he_patch_root, "matched_HE_slides_data", he_case_name),
                os.path.join(he_patch_root, f"{project_name}_matched_HE_slides_data", he_case_name),
            ]
        )
    he_roots.extend(
        [
            os.path.join(root, "matched_HE_patches", he_case_name),
            os.path.join(root, "patch_level_HE_channels", he_case_name),
        ]
    )
    ihc_roots = []
    for ihc_patch_root in ihc_patch_roots:
        ihc_roots.extend(
            [
                os.path.join(ihc_patch_root, f"{ihc_project}_slides_data", ihc_case_name),
                os.path.join(ihc_patch_root, f"{ihc_project}_IHC_slides_data", ihc_case_name),
                os.path.join(ihc_patch_root, f"{project_name}_slides_data", ihc_case_name),
                os.path.join(ihc_patch_root, f"{project_name}_IHC_slides_data", ihc_case_name),
            ]
        )
    ihc_roots.append(os.path.join(root, "patches_IHC", ihc_case_name))
    dab_roots = [
        os.path.join(ihc_root, "patch_level_DAB_channels", ihc_case_name),
        os.path.join(root, "patch_level_DAB_channels", ihc_case_name),
    ]
    return he_roots, ihc_roots, dab_roots


# -------------------------------------------------------------------------
# Macenko-style stain separation and robust scalar morphology descriptors.
# -------------------------------------------------------------------------


def estimate_stain_vectors_macenko(img, alpha=1.0, beta=0.15, stain_type="he"):
    img_od = -np.log10((img.astype(np.float32) + 1.0) / 255.0)
    pixels_od = img_od.reshape(-1, 3)
    od_magnitude = np.linalg.norm(pixels_od, axis=1)
    pixels_od_filtered = pixels_od[od_magnitude > beta]

    stain_type = str(stain_type).lower()
    if pixels_od_filtered.shape[0] < 32:
        if stain_type in ["ihc", "h-dab", "hdab"]:
            return np.array(
                [[0.268, 0.570, 0.776], [0.650, 0.704, 0.286]],
                dtype=np.float32,
            )
        return np.array(
            [[0.5626, 0.7201, 0.4062], [0.2159, 0.8012, 0.5581]],
            dtype=np.float32,
        )

    _, _, vh = np.linalg.svd(pixels_od_filtered, full_matrices=False)
    principal_components = vh[:2, :].T
    projected_pixels = pixels_od_filtered @ principal_components
    angles = np.arctan2(projected_pixels[:, 1], projected_pixels[:, 0])
    min_angle = np.percentile(angles, alpha)
    max_angle = np.percentile(angles, 100 - alpha)

    v1 = principal_components @ np.array([np.cos(min_angle), np.sin(min_angle)])
    v2 = principal_components @ np.array([np.cos(max_angle), np.sin(max_angle)])
    v1 = v1 / (np.linalg.norm(v1) + 1e-8)
    v2 = v2 / (np.linalg.norm(v2) + 1e-8)

    # Put hematoxylin-like vector first for HE. For IHC we keep the convention
    # used by the user's DAB unit: channel 0 is DAB-like, channel 1 is H-like.
    h_like_first = v1[2] >= v1[0]
    if stain_type in ["ihc", "h-dab", "hdab"]:
        if h_like_first:
            return np.stack([v2, v1]).astype(np.float32)
        return np.stack([v1, v2]).astype(np.float32)

    if h_like_first:
        return np.stack([v1, v2]).astype(np.float32)
    return np.stack([v2, v1]).astype(np.float32)


def color_deconvolution_macenko(img, stain_type="he"):
    stain_vectors = estimate_stain_vectors_macenko(img, stain_type=stain_type)
    img_od = -np.log10((img.astype(np.float32) + 1.0) / 255.0)
    pixels_od = img_od.reshape(-1, 3)

    try:
        # Exact vectorized NNLS for a two-vector stain basis. Candidates are
        # the feasible unconstrained solution, either one-vector boundary,
        # and the origin; the minimum-residual candidate is selected per pixel.
        basis = stain_vectors.T.astype(np.float32)  # [3, 2]
        unconstrained = pixels_od @ np.linalg.pinv(stain_vectors)
        candidate_both = np.clip(unconstrained, 0.0, None)

        a0 = basis[:, 0]
        a1 = basis[:, 1]
        c0 = np.clip((pixels_od @ a0) / (float(a0 @ a0) + 1e-8), 0.0, None)
        c1 = np.clip((pixels_od @ a1) / (float(a1 @ a1) + 1e-8), 0.0, None)
        candidate_0 = np.stack([c0, np.zeros_like(c0)], axis=1)
        candidate_1 = np.stack([np.zeros_like(c1), c1], axis=1)
        candidate_zero = np.zeros_like(candidate_both)

        candidates = np.stack(
            [candidate_both, candidate_0, candidate_1, candidate_zero],
            axis=1,
        )
        recon = candidates @ basis.T
        residual = np.square(recon - pixels_od[:, None, :]).sum(axis=2)
        best = np.argmin(residual, axis=1)
        concentrations = candidates[np.arange(len(candidates)), best]
        concentrations = concentrations.reshape(img.shape[0], img.shape[1], 2)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Stain concentration decomposition failed") from exc

    concentrations_norm = np.zeros_like(concentrations, dtype=np.uint8)
    for i in range(2):
        channel = concentrations[:, :, i].copy()
        channel[channel < 0] = 0
        cmin = float(np.min(channel))
        cmax = float(np.max(channel))
        if cmax > cmin:
            channel = (channel - cmin) / (cmax - cmin + 1e-8) * 255.0
        concentrations_norm[:, :, i] = (255.0 - channel.astype(np.uint8)).astype(np.uint8)
    return concentrations_norm, concentrations, stain_vectors


def channel_descriptor(stain_intensity):
    arr = np.asarray(stain_intensity, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    gray = (arr * 255.0).astype(np.uint8)

    edge = cv2.Canny(gray, 50, 150)
    high_ratio = float(np.mean(arr > 0.35))
    edge_density = float(np.mean(edge > 0))
    return np.array(
        [
            float(arr.mean()),
            float(arr.std()),
            float(np.percentile(arr, 75)),
            high_ratio,
            edge_density,
        ],
        dtype=np.float32,
    )


def zero_morph():
    return np.zeros((5,), dtype=np.float32)


def zero_hematoxylin_map(size):
    return np.zeros((int(size), int(size)), dtype=np.float32)


def normalize_stain_map(stain_intensity, size):
    arr = np.asarray(stain_intensity, dtype=np.float32)
    if arr.max() > 1.5:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return cv2.resize(arr, (int(size), int(size)), interpolation=cv2.INTER_AREA).astype(np.float32)


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def fast_hematoxylin_proxy(image):
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    rgb = small.astype(np.float32) / 255.0
    blue_excess = rgb[:, :, 2] - 0.5 * rgb[:, :, 0] - 0.2 * rgb[:, :, 1]
    inv_gray = 1.0 - gray
    proxy = 0.6 * inv_gray + 0.4 * np.clip(blue_excess, 0.0, 1.0)
    proxy = np.clip(proxy, 0.0, 1.0)
    return proxy


def fast_dab_proxy(image):
    small = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    rgb = small.astype(np.float32) / 255.0
    # Brown DAB proxy: stronger when red/green absorb more than blue.
    brown = 0.55 * rgb[:, :, 0] + 0.35 * rgb[:, :, 1] - 0.45 * rgb[:, :, 2]
    brown = brown - float(brown.min())
    if float(brown.max()) > 0:
        brown = brown / float(brown.max())
    return np.clip(brown, 0.0, 1.0)


def compute_he_morphology(he_path, mode="macenko", map_size=32, scalar_mode="legacy_patch_mean"):
    if scalar_mode == "train_standardized":
        if mode != "macenko" or not he_path or not os.path.isfile(he_path):
            raise ValueError("Raw H concentration requires a Macenko RGB H&E patch")
        norm, raw, _ = color_deconvolution_macenko(load_rgb(he_path), stain_type="he")
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Invalid H&E concentration map")
        h_channel = 255.0 - norm[:, :, 0].astype(np.float32)
        return float(raw[:, :, 0].mean()), channel_descriptor(h_channel), normalize_stain_map(h_channel, map_size)
    if mode == "none":
        return 0.0, zero_morph(), zero_hematoxylin_map(map_size)
    if not he_path or not os.path.exists(he_path):
        return 0.0, zero_morph(), zero_hematoxylin_map(map_size)
    try:
        image = load_rgb(he_path)
        if mode == "macenko":
            concentrations_norm, _, _ = color_deconvolution_macenko(image, stain_type="he")
            h_channel = 255.0 - concentrations_norm[:, :, 0].astype(np.float32)
        else:
            h_channel = fast_hematoxylin_proxy(image)
        h_desc = channel_descriptor(h_channel)
        return float(h_desc[0]), h_desc, normalize_stain_map(h_channel, map_size)
    except Exception:
        return 0.0, zero_morph(), zero_hematoxylin_map(map_size)


def compute_ihc_morphology(ihc_path, dab_path, mode="macenko", map_size=32, scalar_mode="legacy_patch_mean"):
    if scalar_mode == "train_standardized":
        if mode != "macenko" or not ihc_path or not os.path.isfile(ihc_path):
            raise ValueError("Raw H/DAB concentrations require a Macenko RGB IHC patch")
        norm, raw, _ = color_deconvolution_macenko(load_rgb(ihc_path), stain_type="ihc")
        if not np.isfinite(raw).all() or np.any(raw < 0):
            raise ValueError("Invalid IHC concentration map")
        h_channel = 255.0 - norm[:, :, 1].astype(np.float32)
        # Never replace raw concentrations with independently normalized DAB PNGs.
        return float(raw[:, :, 1].mean()), float(raw[:, :, 0].mean()), channel_descriptor(h_channel), normalize_stain_map(h_channel, map_size)
    h_desc = zero_morph()
    h_map = zero_hematoxylin_map(map_size)
    dab_mean = 0.0

    if mode == "none":
        return 0.0, 0.0, h_desc, h_map

    if ihc_path and os.path.exists(ihc_path):
        try:
            image = load_rgb(ihc_path)
            if mode == "macenko":
                concentrations_norm, _, _ = color_deconvolution_macenko(image, stain_type="ihc")
                # User's IHC DAB unit uses 255 - concentrations_norm[:, :, 0].
                dab_channel = 255.0 - concentrations_norm[:, :, 0].astype(np.float32)
                h_channel = 255.0 - concentrations_norm[:, :, 1].astype(np.float32)
            else:
                h_channel = fast_hematoxylin_proxy(image)
                dab_channel = fast_dab_proxy(image)
            h_desc = channel_descriptor(h_channel)
            h_map = normalize_stain_map(h_channel, map_size)
            dab_mean = float(channel_descriptor(dab_channel)[0])
        except Exception:
            pass

    if dab_path and os.path.exists(dab_path):
        try:
            dab_gray = np.array(Image.open(dab_path).convert("L"), dtype=np.float32)
            # Precomputed user DAB channel is already higher for stronger DAB.
            dab_mean = float(np.clip(dab_gray / 255.0, 0.0, 1.0).mean())
        except Exception:
            pass

    return float(h_desc[0]), float(dab_mean), h_desc, h_map


def get_tile_feature_dir(base_path, project_name, pair_case_name):
    candidates = [
        os.path.join(base_path, project_name, "FusionFeatures_patch_level_tiles", pair_case_name),
        os.path.join(base_path, get_he_project_name(project_name), "FusionFeatures_patch_level_tiles", pair_case_name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return candidates[0]


def list_tile_feature_paths(base_path, project_name, pair_case_name):
    feature_dir = get_tile_feature_dir(base_path, project_name, pair_case_name)
    if not os.path.isdir(feature_dir):
        raise FileNotFoundError(f"Missing tile feature directory: {feature_dir}")
    return sorted(
        [
            os.path.join(feature_dir, name)
            for name in os.listdir(feature_dir)
            if name.lower().endswith(".pt")
        ]
    )


def get_case_cache_path(row, args, cache_dir, project_name, pair_case_name):
    scalar_mode = getattr(args, "stain_scalar_mode", "train_standardized")
    cache_tag = f"paper_v2_{scalar_mode}_ps{args.patch_size}_ssim{args.ssim_map_size}_{args.morphology_mode}"
    feature_source = str(getattr(args, "feature_source", "fusion_tiles")).lower()
    if feature_source == "wsi_pt":
        encoder_name = safe_name(getattr(args, "image_encoder_name", "wsi_encoder"))
        return os.path.join(
            cache_dir,
            f"{cache_tag}_wsi_pt_{encoder_name}",
            project_name,
            safe_name(pair_case_name) + ".pt",
        )
    return os.path.join(cache_dir, cache_tag, project_name, safe_name(pair_case_name) + ".pt")


def build_or_load_case_cache(row, args, cache_dir):
    pair_case_name = str(row["pair_case_name"])
    label = int(row["label"])
    case_id = str(row.get("case_id", pair_case_name))
    project_name = str(row.get("project_name", args.project_name))
    cache_path = get_case_cache_path(row, args, cache_dir, project_name, pair_case_name)

    if os.path.exists(cache_path) and not args.force_rebuild_cache:
        return torch.load(cache_path, map_location="cpu")

    he_case_name, ihc_case_name = parse_pair_case_name(pair_case_name)
    he_file = str(row.get("he_file", he_case_name))
    ihc_file = str(row.get("ihc_file", ihc_case_name))
    he_case_name = strip_slide_ext(he_file or he_case_name)
    ihc_case_name = strip_slide_ext(ihc_file or ihc_case_name)
    feature_paths = list_tile_feature_paths(args.base_path, project_name, pair_case_name)

    he_roots, ihc_roots, dab_roots = get_patch_roots(
        args.base_path,
        project_name,
        he_case_name,
        ihc_case_name,
        patch_size=args.patch_size,
    )
    if "he_patches_path" in row and pd.notna(row.get("he_patches_path")):
        he_roots = [str(row.get("he_patches_path"))] + he_roots
    if "ihc_patches_path" in row and pd.notna(row.get("ihc_patches_path")):
        ihc_roots = [str(row.get("ihc_patches_path"))] + ihc_roots

    feature_source = str(getattr(args, "feature_source", "fusion_tiles")).lower()
    if feature_source == "wsi_pt":
        he_wsi_path, ihc_wsi_path = get_wsi_feature_paths(
            pair_case_name,
            project_name,
            args.image_encoder_name,
            args,
            he_file=he_file,
            ihc_file=ihc_file,
        )
        if not os.path.exists(he_wsi_path):
            raise FileNotFoundError(f"Missing HE WSI feature: {he_wsi_path}")
        if not os.path.exists(ihc_wsi_path):
            raise FileNotFoundError(f"Missing IHC WSI feature: {ihc_wsi_path}")
        he_wsi_features = load_wsi_feature_matrix(he_wsi_path)
        ihc_wsi_features = load_wsi_feature_matrix(ihc_wsi_path)
        n_available = min(len(feature_paths), he_wsi_features.shape[0], ihc_wsi_features.shape[0])
        if n_available <= 0:
            raise RuntimeError(f"No usable WSI features found for case {pair_case_name}")
        if n_available != len(feature_paths) or n_available != he_wsi_features.shape[0] or n_available != ihc_wsi_features.shape[0]:
            msg = (
                f"[INFO] {pair_case_name}: using first {n_available} patches "
                f"(tiles={len(feature_paths)}, HE={he_wsi_features.shape[0]}, IHC={ihc_wsi_features.shape[0]})"
            )
            if args.allow_wsi_feature_truncate:
                print(msg)
            else:
                raise RuntimeError(msg)
        feature_paths = feature_paths[:n_available]
    else:
        he_wsi_features = None
        ihc_wsi_features = None

    he_features = []
    ihc_features = []
    tile_names = []
    he_h_values = []
    ihc_h_values = []
    ihc_dab_values = []
    he_morph_values = []
    ihc_morph_values = []
    he_h_maps = []
    ihc_h_maps = []
    he_patch_paths = []
    ihc_patch_paths = []
    dab_patch_paths = []

    for patch_idx, feature_path in enumerate(feature_paths):
        tile_name = os.path.splitext(os.path.basename(feature_path))[0]
        if feature_source == "wsi_pt":
            he_feature = he_wsi_features[patch_idx].reshape(-1).float()
            ihc_feature = ihc_wsi_features[patch_idx].reshape(-1).float()
        else:
            try:
                feature_tensor = torch_load_any(feature_path)
                he_feature, ihc_feature = split_he_ihc_feature(feature_tensor, args.feature_dim)
            except Exception as exc:
                print(f"[WARNING] Skip feature {feature_path}: {exc}")
                continue

        he_patch_path = find_image_file(he_roots, tile_name)
        ihc_patch_path = find_image_file(ihc_roots, tile_name)
        dab_patch_path = find_image_file(dab_roots, tile_name)

        needs_morphology = (
            args.input_mode == "multistain"
            and args.alignment_mode in {"morph", "both"}
            and args.morphology_mode != "none"
        )
        if needs_morphology and not args.allow_missing_morphology:
            missing_patch = []
            if not he_patch_path:
                missing_patch.append("H&E")
            if not ihc_patch_path:
                missing_patch.append("IHC")
            if missing_patch:
                raise FileNotFoundError(
                    f"Missing {'/'.join(missing_patch)} patch image for {pair_case_name}/{tile_name}; "
                    "SSIM alignment requires both hematoxylin images. Check --patch_size and patch roots."
                )

        he_h, he_morph, he_h_map = compute_he_morphology(
            he_patch_path,
            mode=args.morphology_mode,
            map_size=args.ssim_map_size,
            scalar_mode=getattr(args, "stain_scalar_mode", "train_standardized"),
        )
        ihc_h, ihc_dab, ihc_morph, ihc_h_map = compute_ihc_morphology(
            ihc_patch_path,
            dab_patch_path,
            mode=args.morphology_mode,
            map_size=args.ssim_map_size,
            scalar_mode=getattr(args, "stain_scalar_mode", "train_standardized"),
        )

        he_features.append(he_feature)
        ihc_features.append(ihc_feature)
        tile_names.append(tile_name)
        he_h_values.append(he_h)
        ihc_h_values.append(ihc_h)
        ihc_dab_values.append(ihc_dab)
        he_morph_values.append(he_morph)
        ihc_morph_values.append(ihc_morph)
        he_h_maps.append(he_h_map)
        ihc_h_maps.append(ihc_h_map)
        he_patch_paths.append(he_patch_path)
        ihc_patch_paths.append(ihc_patch_path)
        dab_patch_paths.append(dab_patch_path)

    if not he_features:
        raise RuntimeError(f"No usable features found for case {pair_case_name}")

    case_obj = {
        "stain_scalar_schema": SCALAR_SCHEMA if getattr(args, "stain_scalar_mode", "train_standardized") == "train_standardized" else "legacy_patch_mean",
        "case_id": case_id,
        "project_name": project_name,
        "pair_case_name": pair_case_name,
        "feature_source": feature_source,
        "image_encoder_name": str(getattr(args, "image_encoder_name", "fusion_tiles")),
        "he_case_name": he_case_name,
        "ihc_case_name": ihc_case_name,
        "he_file": he_file,
        "ihc_file": ihc_file,
        "label": label,
        "tile_names": tile_names,
        "he_features": torch.stack(he_features, dim=0).float(),
        "ihc_features": torch.stack(ihc_features, dim=0).float(),
        "he_h": torch.tensor(he_h_values, dtype=torch.float32),
        "ihc_h": torch.tensor(ihc_h_values, dtype=torch.float32),
        "ihc_dab": torch.tensor(ihc_dab_values, dtype=torch.float32),
        "he_morph": torch.tensor(np.stack(he_morph_values, axis=0), dtype=torch.float32),
        "ihc_morph": torch.tensor(np.stack(ihc_morph_values, axis=0), dtype=torch.float32),
        "he_h_map": torch.tensor(np.stack(he_h_maps, axis=0), dtype=torch.float32),
        "ihc_h_map": torch.tensor(np.stack(ihc_h_maps, axis=0), dtype=torch.float32),
        "he_patch_paths": he_patch_paths,
        "ihc_patch_paths": ihc_patch_paths,
        "dab_patch_paths": dab_patch_paths,
    }
    ensure_dir(os.path.dirname(cache_path))
    torch.save(case_obj, cache_path)

    csv_path = os.path.join(
        os.path.dirname(cache_path),
        safe_name(pair_case_name) + "_stain_morphology.csv",
    )
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "tile_name",
                "he_h_mean",
                "ihc_h_mean",
                "ihc_dab_mean",
                "he_patch_path",
                "ihc_patch_path",
                "dab_patch_path",
            ],
        )
        writer.writeheader()
        for i, tile_name in enumerate(tile_names):
            writer.writerow(
                {
                    "tile_name": tile_name,
                    "he_h_mean": he_h_values[i],
                    "ihc_h_mean": ihc_h_values[i],
                    "ihc_dab_mean": ihc_dab_values[i],
                    "he_patch_path": he_patch_paths[i],
                    "ihc_patch_path": ihc_patch_paths[i],
                    "dab_patch_path": dab_patch_paths[i],
                }
            )

    return case_obj


def enrich_metadata_columns(df, project_name):
    df = df.copy()
    df["project_name"] = project_name
    if "case_id" not in df.columns:
        df["case_id"] = df["pair_case_name"]
    if "he_file" not in df.columns:
        df["he_file"] = df["pair_case_name"].apply(lambda x: parse_pair_case_name(x)[0])
    if "ihc_file" not in df.columns:
        df["ihc_file"] = df["pair_case_name"].apply(lambda x: parse_pair_case_name(x)[1])
    df["he_stem"] = df["he_file"].apply(strip_slide_ext)
    df["ihc_stem"] = df["ihc_file"].apply(strip_slide_ext)
    df["pair_key"] = df.apply(lambda r: normalize_pair_key(r["he_file"], r["ihc_file"]), axis=1)
    df["pair_key_rev"] = df.apply(lambda r: normalize_pair_key(r["ihc_file"], r["he_file"]), axis=1)
    df["case_pair_key"] = (
        df["case_id"].astype(str).str.lower()
        + "::"
        + df["pair_key"].astype(str)
    )
    return df


def deduplicate_case_metadata(df, project_name):
    before = len(df)
    dedupe_cols = [col for col in ["case_id", "ihc_file", "he_file"] if col in df.columns]
    if len(dedupe_cols) == 3:
        df = df.drop_duplicates(subset=dedupe_cols, keep="first")
    else:
        df = df.drop_duplicates(subset=["pair_case_name"], keep="first")
    df = df.drop_duplicates(subset=["pair_case_name"], keep="first").reset_index(drop=True)
    removed = before - len(df)
    if removed > 0:
        print(f"[INFO] Removed {removed} duplicated pair rows from {project_name}.")
    return df


def find_metadata_path(project_name, base_path):
    candidates = [
        os.path.join(base_path, project_name, "FusionFeatures_patch_level_tiles", "pair_labels_with_case_id.csv"),
        os.path.join(base_path, get_he_project_name(project_name), "FusionFeatures_patch_level_tiles", "pair_labels_with_case_id.csv"),
        os.path.join(base_path, project_name, "pair_labels_with_case_id.csv"),
        os.path.join(base_path, get_he_project_name(project_name), "pair_labels_with_case_id.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError("Missing metadata. Tried:\n" + "\n".join(candidates))


def row_has_required_features(row, args, project_name):
    pair_case_name = str(row["pair_case_name"])
    try:
        tile_paths = list_tile_feature_paths(args.base_path, project_name, pair_case_name)
    except Exception as exc:
        return False, f"missing old tile dir or no tile pt: {exc}"
    if len(tile_paths) == 0:
        return False, "old tile dir has no tile pt files"

    feature_source = str(getattr(args, "feature_source", "fusion_tiles")).lower()
    if feature_source != "wsi_pt":
        return True, ""

    he_path, ihc_path = get_wsi_feature_paths(
        pair_case_name,
        project_name,
        args.image_encoder_name,
        args,
        he_file=str(row.get("he_file", "")),
        ihc_file=str(row.get("ihc_file", "")),
    )
    missing = []
    if not os.path.exists(he_path):
        missing.append(f"missing HE WSI feature: {he_path}")
    if not os.path.exists(ihc_path):
        missing.append(f"missing IHC WSI feature: {ihc_path}")
    if missing:
        return False, " | ".join(missing)
    return True, ""


def load_case_metadata(project_name, base_path, expected_cases=0, args=None):
    metadata_path = find_metadata_path(project_name, base_path)
    df = pd.read_csv(metadata_path)
    required = {"pair_case_name", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Metadata missing columns: {missing}")
    df = enrich_metadata_columns(df, project_name)
    df = deduplicate_case_metadata(df, project_name)

    ok_rows = []
    missing_examples = []
    for _, row in df.iterrows():
        if args is None:
            feature_dir = get_tile_feature_dir(base_path, project_name, str(row["pair_case_name"]))
            ok = os.path.isdir(feature_dir)
            reason = f"Missing feature dir, skipped: {feature_dir}"
        else:
            ok, reason = row_has_required_features(row, args, project_name)
        if ok:
            ok_rows.append(row)
        else:
            if len(missing_examples) < 10:
                missing_examples.append((str(row["pair_case_name"]), reason))
    if not ok_rows:
        raise RuntimeError("No cases with existing feature directories.")
    out = pd.DataFrame(ok_rows).reset_index(drop=True)
    print(f"[INFO] Loaded {len(out)} cases from {project_name}")
    if missing_examples:
        print("[WARNING] Missing feature examples, up to 10:")
        for item in missing_examples:
            print(f"  - {item[0]}: {item[1]}")
    if expected_cases > 0 and len(out) != expected_cases:
        print(
            f"[WARNING] {project_name} expected {expected_cases} cases after dedupe, "
            f"but loaded {len(out)} cases with existing feature dirs."
        )
    return out


def split_csv_default_path(project_name, base_path, file_name):
    return os.path.join(
        base_path,
        project_name,
        "FusionFeatures_patch_level_tiles",
        file_name,
    )


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
    df["split_case_pair_key"] = (
        df["case_id"].astype(str).str.lower()
        + "::"
        + df["split_pair_key"].astype(str)
    )
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
        if "he_patches_path" in split_row:
            row["he_patches_path"] = split_row.get("he_patches_path", "")
        if "ihc_patches_path" in split_row:
            row["ihc_patches_path"] = split_row.get("ihc_patches_path", "")
        row["fixed_split"] = split_name
        rows.append(row)

    out = pd.DataFrame(rows).drop_duplicates(subset=["pair_case_name"]).reset_index(drop=True)

    if unmatched:
        unmatched_path = os.path.join(results_dir, "splits", f"{split_name}_unmatched_rows.csv")
        ensure_dir(os.path.dirname(unmatched_path))
        pd.DataFrame(unmatched).to_csv(unmatched_path, index=False)
        print(f"[WARNING] {split_name}: {len(unmatched)} split rows unmatched. Saved: {unmatched_path}")

    print(f"[INFO] Matched fixed {split_name} split: {len(out)}/{len(split_df)} rows")
    return out


def stratified_split_cases(df, test_size, val_size, seed):
    """Split patient groups, then return every image pair for the selected patients."""

    case_labels = df[["case_id", "label"]].copy()
    case_labels["case_id"] = case_labels["case_id"].astype(str)
    inconsistent = case_labels.groupby("case_id")["label"].nunique()
    inconsistent = inconsistent[inconsistent > 1]
    if not inconsistent.empty:
        raise ValueError(
            "Each patient must have one response label. Inconsistent case_id values: "
            + ", ".join(inconsistent.index[:10])
        )
    groups = case_labels.drop_duplicates("case_id").reset_index(drop=True)

    def maybe_stratify(group_df):
        labels = group_df["label"].astype(int)
        return labels if labels.nunique() == 2 and labels.value_counts().min() >= 2 else None

    def select_pairs(group_df):
        selected = set(group_df["case_id"].astype(str))
        return df[df["case_id"].astype(str).isin(selected)].copy()

    if test_size <= 0:
        train_val_groups = groups.copy()
        test_groups = groups.iloc[0:0].copy()
    else:
        train_val_groups, test_groups = train_test_split(
            groups,
            test_size=test_size,
            random_state=seed,
            stratify=maybe_stratify(groups),
        )

    if val_size <= 0 or len(train_val_groups) < 4:
        train_groups = train_val_groups.copy()
        val_groups = train_val_groups.iloc[0:0].copy()
    else:
        train_groups, val_groups = train_test_split(
            train_val_groups,
            test_size=val_size,
            random_state=seed,
            stratify=maybe_stratify(train_val_groups),
        )

    train = select_pairs(train_groups)
    val = select_pairs(val_groups)
    test = select_pairs(test_groups)
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


class CaseMILDataset(Dataset):
    def __init__(self, metadata_df, args, split_name, training=False):
        self.metadata = metadata_df.reset_index(drop=True)
        self.args = args
        self.split_name = split_name
        self.training = training
        self.cache_dir = get_cache_dir(args)

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
        case_obj = build_or_load_case_cache(row, self.args, self.cache_dir)
        if getattr(self.args, "stain_scalar_mode", "train_standardized") == "train_standardized":
            case_obj = scale_case(case_obj, getattr(self.args, "stain_scaler", None))
        n_ihc = int(case_obj["ihc_features"].shape[0])
        keep_ihc = self._sample_indices(n_ihc, idx)
        keep_ihc_t = torch.tensor(keep_ihc, dtype=torch.long)

        return {
            "case_id": case_obj["case_id"],
            "project_name": case_obj["project_name"],
            "pair_case_name": case_obj["pair_case_name"],
            "he_file": case_obj["he_file"],
            "ihc_file": case_obj["ihc_file"],
            "label": torch.tensor(float(case_obj["label"]), dtype=torch.float32),
            "tile_names": [case_obj["tile_names"][i] for i in keep_ihc],
            "ihc_anchor_tile_names": [case_obj["tile_names"][i] for i in keep_ihc],
            "he_candidate_tile_names": list(case_obj["tile_names"]),
            "he_patch_paths": list(case_obj["he_patch_paths"]),
            "ihc_patch_paths": [case_obj["ihc_patch_paths"][i] for i in keep_ihc],
            # The paper samples at most 1,024 IHC anchors while retaining all
            # H&E patches from the case as matching candidates.
            "he_features": case_obj["he_features"],
            "ihc_features": case_obj["ihc_features"][keep_ihc_t],
            "he_h": case_obj["he_h"],
            "ihc_h": case_obj["ihc_h"][keep_ihc_t],
            "ihc_dab": case_obj["ihc_dab"][keep_ihc_t],
            "he_morph": case_obj["he_morph"],
            "ihc_morph": case_obj["ihc_morph"][keep_ihc_t],
            "he_h_map": case_obj["he_h_map"],
            "ihc_h_map": case_obj["ihc_h_map"][keep_ihc_t],
            "ihc_source_indices": keep_ihc_t,
        }


def case_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    if not batch:
        return None
    if len(batch) != 1:
        raise ValueError("Use batch_size=1 for variable-length case-level MIL.")
    return batch[0]


def to_device(sample, device):
    out = dict(sample)
    for key in [
        "label",
        "he_features",
        "ihc_features",
        "he_h",
        "ihc_h",
        "ihc_dab",
        "he_morph",
        "ihc_morph",
        "he_h_map",
        "ihc_h_map",
        "ihc_source_indices",
    ]:
        out[key] = out[key].to(device)
    return out


def get_query_texts():
    return [query_text for _, query_text in DEFAULT_QUERIES]


def resolve_text_device(args):
    if args.query_encoder_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(args.query_encoder_device)
    return torch.device("cpu")


def resolve_hf_dtype(args, text_device):
    dtype_name = str(args.text_encoder_dtype).lower()
    if dtype_name == "auto":
        return torch.float16 if text_device.type == "cuda" else torch.float32
    if dtype_name == "float16":
        return torch.float16
    if dtype_name == "bfloat16":
        return torch.bfloat16
    return torch.float32


def masked_mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).to(last_hidden_state.dtype)
    summed = torch.sum(last_hidden_state * mask, dim=1)
    denom = torch.clamp(mask.sum(dim=1), min=1.0)
    return summed / denom


def last_token_pool(last_hidden_state, attention_mask):
    lengths = torch.clamp(attention_mask.sum(dim=1) - 1, min=0)
    batch_idx = torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device)
    return last_hidden_state[batch_idx, lengths]


def load_tfidf_query_embeddings(args, device):
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
    except Exception as exc:
        raise ImportError("scikit-learn is required for TF-IDF query encoding.") from exc

    texts = get_query_texts()
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, args.tfidf_ngram_max),
        min_df=1,
        norm="l2",
        token_pattern=r"(?u)\b\w+\b",
    )
    embeddings = vectorizer.fit_transform(texts).astype(np.float32).toarray()
    metadata = {
        "text_encoder_type": "tfidf",
        "text_model": "TfidfVectorizer",
        "tfidf_ngram_max": args.tfidf_ngram_max,
        "embedding_dim": int(embeddings.shape[1]),
        "vocabulary": vectorizer.get_feature_names_out().tolist(),
    }
    return torch.from_numpy(embeddings).float().to(device), metadata


def load_clip_query_embeddings(args, device):
    text_device = resolve_text_device(args)
    dtype = resolve_hf_dtype(args, text_device)

    try:
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
    except Exception as exc:
        raise ImportError(
            "transformers with PyTorch backend is required for CLIP text encoding. "
            "If transformers says PyTorch is unavailable, install a transformers version "
            "compatible with your torch version, for example transformers==4.41.2 for torch 2.1."
        ) from exc

    print(f"[INFO] Loading CLIP text encoder: {args.clip_text_model} on {text_device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.clip_text_model,
        local_files_only=args.local_files_only,
    )
    text_model = CLIPTextModelWithProjection.from_pretrained(
        args.clip_text_model,
        local_files_only=args.local_files_only,
        torch_dtype=dtype,
    ).to(text_device)
    text_model.eval()

    batch = tokenizer(
        get_query_texts(),
        padding=True,
        truncation=True,
        max_length=args.text_max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(text_device) for key, value in batch.items()}

    with torch.no_grad():
        output = text_model(**batch)
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            embeddings = output.text_embeds
        elif hasattr(output, "pooler_output") and output.pooler_output is not None:
            embeddings = output.pooler_output
        else:
            embeddings = output.last_hidden_state[:, 0]

    embeddings = embeddings.detach().float().to(device)
    metadata = {
        "text_encoder_type": "clip",
        "text_model": args.clip_text_model,
        "embedding_dim": int(embeddings.shape[1]),
        "text_max_length": args.text_max_length,
        "text_encoder_device": str(text_device),
        "text_encoder_dtype": str(dtype).replace("torch.", ""),
    }
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, metadata


def load_qwen_query_embeddings(args, device):
    text_device = resolve_text_device(args)
    dtype = resolve_hf_dtype(args, text_device)

    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "transformers with PyTorch backend is required for Qwen query encoding. "
            "If transformers says PyTorch is unavailable, install a transformers version "
            "compatible with your torch version, for example transformers==4.41.2 for torch 2.1."
        ) from exc

    print(f"[INFO] Loading Qwen decoder-only text encoder: {args.qwen_text_model} on {text_device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.qwen_text_model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    text_model = AutoModel.from_pretrained(
        args.qwen_text_model,
        local_files_only=args.local_files_only,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=dtype,
    ).to(text_device)
    text_model.eval()

    batch = tokenizer(
        get_query_texts(),
        padding=True,
        truncation=True,
        max_length=args.text_max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(text_device) for key, value in batch.items()}

    with torch.no_grad():
        output = text_model(**batch, use_cache=False)
        hidden = output.last_hidden_state
        if args.qwen_pooling == "mean":
            embeddings = masked_mean_pool(hidden, batch["attention_mask"])
        else:
            embeddings = last_token_pool(hidden, batch["attention_mask"])

    embeddings = embeddings.detach().float().to(device)
    metadata = {
        "text_encoder_type": "qwen2_5_0_5b",
        "text_model": args.qwen_text_model,
        "embedding_dim": int(embeddings.shape[1]),
        "pooling": args.qwen_pooling,
        "text_max_length": args.text_max_length,
        "text_encoder_device": str(text_device),
        "text_encoder_dtype": str(dtype).replace("torch.", ""),
    }
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, metadata


def load_clinicalbert_query_embeddings(args, device):
    text_device = resolve_text_device(args)

    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "transformers is required for ClinicalBERT query encoding, or pass "
            "--query_embedding_path to use precomputed query embeddings."
        ) from exc

    print(f"[INFO] Loading ClinicalBERT query encoder: {args.text_model} on {text_device}")
    tokenizer = AutoTokenizer.from_pretrained(
        args.text_model,
        local_files_only=args.local_files_only,
    )
    text_model = AutoModel.from_pretrained(
        args.text_model,
        local_files_only=args.local_files_only,
    ).to(text_device)
    text_model.eval()

    batch = tokenizer(
        get_query_texts(),
        padding=True,
        truncation=True,
        max_length=args.text_max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(text_device) for key, value in batch.items()}

    with torch.no_grad():
        output = text_model(**batch)
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            embeddings = output.pooler_output
        else:
            embeddings = output.last_hidden_state[:, 0]

    embeddings = embeddings.detach().float().to(device)
    metadata = {
        "text_encoder_type": "clinicalbert",
        "text_model": args.text_model,
        "embedding_dim": int(embeddings.shape[1]),
        "text_max_length": args.text_max_length,
        "text_encoder_device": str(text_device),
    }
    del text_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return embeddings, metadata


def load_query_embeddings(args, device):
    if args.query_embedding_path:
        arr = np.load(args.query_embedding_path)
        emb = torch.as_tensor(arr, dtype=torch.float32)
        if emb.ndim != 2 or emb.shape[0] != len(DEFAULT_QUERIES):
            raise ValueError(
                f"Expected query embeddings shape [6, d], got {tuple(emb.shape)}"
            )
        metadata = {
            "text_encoder_type": "precomputed",
            "query_embedding_path": args.query_embedding_path,
            "embedding_dim": int(emb.shape[1]),
        }
        return emb.to(device), metadata

    text_encoder_type = str(args.text_encoder_type).lower()
    if text_encoder_type == "tfidf":
        return load_tfidf_query_embeddings(args, device)
    if text_encoder_type == "clip":
        return load_clip_query_embeddings(args, device)
    if text_encoder_type in ["qwen", "qwen2.5-0.5b", "qwen2_5_0_5b"]:
        return load_qwen_query_embeddings(args, device)
    if text_encoder_type == "clinicalbert":
        return load_clinicalbert_query_embeddings(args, device)
    raise ValueError(f"Unsupported text_encoder_type: {args.text_encoder_type}")


def make_mlp(in_dim, hidden_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, out_dim),
    )


class ThreeLayerGatedMLP(nn.Module):
    """Three-layer gated classifier used after concatenating query evidence."""

    def __init__(self, in_dim, hidden_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(in_dim)
        self.fc1 = nn.Linear(in_dim, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        value, gate = self.fc1(x).chunk(2, dim=-1)
        x = F.gelu(value) * torch.sigmoid(gate)
        x = self.dropout(x)
        x = self.dropout(F.gelu(self.fc2(x)))
        return self.fc3(x)


class VGMS(nn.Module):
    def __init__(
        self,
        query_embeddings,
        feature_dim=768,
        morph_dim=5,
        hidden_dim=256,
        align_dim=256,
        dropout=0.35,
        align_topk=64,
        temperature=1.0,
        disable_morph_alignment=False,
        disable_query_guidance=False,
        disable_query_branch=False,
        disable_alignment=False,
        input_mode="multistain",
        alignment_mode="both",
        alignment_lambda=0.5,
        spatial_gamma=0.25,
        spatial_sigma=8.0,
        patch_stride=250.0,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.align_topk = int(align_topk)
        self.temperature = float(temperature)
        self.disable_morph_alignment = bool(disable_morph_alignment)
        self.disable_query_guidance = bool(disable_query_guidance)
        self.disable_query_branch = bool(disable_query_branch)
        self.disable_alignment = bool(disable_alignment)
        self.input_mode = str(input_mode).lower()
        self.alignment_mode = str(alignment_mode).lower()
        self.alignment_lambda = float(alignment_lambda)
        self.spatial_gamma = float(spatial_gamma)
        self.spatial_sigma = float(spatial_sigma)
        self.patch_stride = float(patch_stride)
        if not 0.0 <= self.alignment_lambda <= 1.0:
            raise ValueError("alignment_lambda must be in [0, 1]")
        if self.align_topk <= 0 or self.temperature <= 0:
            raise ValueError("align_topk and align_temperature must be positive")
        if self.spatial_gamma < 0 or self.spatial_sigma <= 0 or self.patch_stride <= 0:
            raise ValueError("spatial_gamma must be nonnegative; spatial_sigma and patch_stride must be positive")
        if self.disable_alignment:
            self.alignment_mode = "none"
        if self.disable_morph_alignment and self.alignment_mode == "both":
            self.alignment_mode = "feature"
        if self.input_mode in ["he", "ihc"]:
            self.alignment_mode = "none"

        self.register_buffer("query_embeddings", query_embeddings.detach().float())
        query_dim = int(query_embeddings.shape[1])
        num_queries = int(query_embeddings.shape[0])
        self.num_queries = num_queries

        # Equation (5): concatenate matched H&E, IHC, H intensity, and DAB
        # intensity before projection into the visual-token space.
        self.visual_proj = make_mlp(feature_dim * 2 + 2, hidden_dim, hidden_dim, dropout)
        self.query_proj = make_mlp(query_dim, hidden_dim, hidden_dim, dropout)
        self.query_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.visual_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        # Used only by the no-query ablation. The full VGMS path below remains
        # the six-query cross-attention described in the paper.
        self.mil_attention_v = nn.Linear(hidden_dim, hidden_dim)
        self.mil_attention_u = nn.Linear(hidden_dim, hidden_dim)
        self.mil_attention_w = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = ThreeLayerGatedMLP(hidden_dim * num_queries, hidden_dim * 2, dropout)

    @staticmethod
    def _pairwise_ssim(ihc_h_map, he_h_map):
        """Global standard-form SSIM for every IHC-H&E hematoxylin pair."""
        x = ihc_h_map.flatten(1).float()
        y = he_h_map.flatten(1).float()
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)
        x_centered = x - x_mean
        y_centered = y - y_mean
        denom = max(int(x.shape[1]) - 1, 1)
        covariance = (x_centered @ y_centered.t()) / denom
        x_var = (x_centered.square().sum(dim=1, keepdim=True) / denom)
        y_var = (y_centered.square().sum(dim=1, keepdim=True) / denom).t()
        c1 = 0.01 ** 2
        c2 = 0.03 ** 2
        luminance = (2.0 * (x_mean @ y_mean.t()) + c1) / (
            x_mean.square() + y_mean.t().square() + c1
        )
        contrast_structure = (2.0 * covariance + c2) / (x_var + y_var + c2)
        return luminance * contrast_structure

    def _similarity(self, he_features, ihc_features, he_h_map, ihc_h_map):
        # Rows are IHC anchors; columns are all H&E candidates from the case.
        s_feat = F.normalize(ihc_features, dim=1) @ F.normalize(he_features, dim=1).t()

        if self.alignment_mode == "feature":
            s_morph = torch.zeros_like(s_feat)
            mix_weights = torch.tensor([1.0, 0.0], device=he_features.device)
            return s_feat, s_morph, s_feat, mix_weights

        s_morph = self._pairwise_ssim(ihc_h_map, he_h_map)

        if self.alignment_mode == "morph":
            mix_weights = torch.tensor([0.0, 1.0], device=he_features.device)
            return s_feat, s_morph, s_morph, mix_weights

        mix_weights = torch.tensor(
            [1.0 - self.alignment_lambda, self.alignment_lambda],
            dtype=he_features.dtype,
            device=he_features.device,
        )
        s_align = mix_weights[0] * s_feat + mix_weights[1] * s_morph
        return s_feat, s_morph, s_align, mix_weights

    def _align(self, he_features, ihc_features, he_h_map, ihc_h_map, source_indices,
               he_tile_names=None, ihc_tile_names=None):
        n_he = int(he_features.shape[0])
        n_ihc = int(ihc_features.shape[0])

        if self.disable_alignment or self.alignment_mode == "none":
            idx = source_indices.long().clamp(max=max(n_he - 1, 0)).reshape(-1, 1)
            aligned = he_features[idx[:, 0]]
            weight = torch.ones(n_ihc, 1, device=he_features.device)
            mix_weights = torch.tensor([1.0, 0.0], device=he_features.device)
            return aligned, idx, weight, None, None, None, None, None, None, mix_weights

        if he_tile_names is None or ihc_tile_names is None:
            raise ValueError("Spiral matching requires H&E and IHC y_x tile names")
        if len(he_tile_names) != n_he or len(ihc_tile_names) != n_ihc or n_he == 0:
            raise ValueError("Tile names must correspond one-to-one to nonempty feature rows")

        he_yx = [tile_coordinate_yx(name) for name in he_tile_names]
        ihc_yx = [tile_coordinate_yx(name) for name in ihc_tile_names]
        return bounded_spiral_align(
            self, he_features, ihc_features, he_h_map, ihc_h_map,
            he_yx, ihc_yx, spiral_candidate_order,
        )

    def forward(self, sample):
        he_features = sample["he_features"]
        ihc_features = sample["ihc_features"]
        he_h = sample["he_h"].float()
        ihc_h = sample["ihc_h"].float()
        ihc_dab = sample["ihc_dab"].float()
        he_h_map = sample["he_h_map"].float().clamp(0.0, 1.0)
        ihc_h_map = sample["ihc_h_map"].float().clamp(0.0, 1.0)
        source_indices = sample["ihc_source_indices"]

        zero_align = torch.tensor([0.0, 0.0], device=he_features.device)
        if self.input_mode == "he":
            n = int(he_features.shape[0])
            matched_he = he_features
            visual_ihc = torch.zeros_like(he_features)
            matched_he_h = he_h
            visual_dab = torch.zeros(n, device=he_features.device)
            align_idx = torch.zeros(n, 1, dtype=torch.long, device=he_features.device)
            align_weights = torch.ones(n, 1, device=he_features.device)
            s_feat = s_morph = s_base = s_align = None
            pairwise_distance = spatial_penalty = None
            align_mix = zero_align
        elif self.input_mode == "ihc":
            n = int(ihc_features.shape[0])
            matched_he = torch.zeros_like(ihc_features)
            visual_ihc = ihc_features
            matched_he_h = ihc_h
            visual_dab = ihc_dab
            align_idx = torch.arange(n, device=ihc_features.device).unsqueeze(1)
            align_weights = torch.ones(n, 1, device=ihc_features.device)
            s_feat = s_morph = s_base = s_align = None
            pairwise_distance = spatial_penalty = None
            align_mix = zero_align
        else:
            (
                matched_he,
                align_idx,
                align_weights,
                s_feat,
                s_morph,
                s_base,
                s_align,
                pairwise_distance,
                spatial_penalty,
                align_mix,
            ) = self._align(
                he_features,
                ihc_features,
                he_h_map,
                ihc_h_map,
                source_indices,
                sample.get("he_candidate_tile_names"),
                sample.get("ihc_anchor_tile_names"),
            )
            selected_he_h = he_h[align_idx]
            matched_he_h = torch.sum(
                align_weights.to(selected_he_h.dtype) * selected_he_h,
                dim=1,
            )
            visual_ihc = ihc_features
            visual_dab = ihc_dab

        visual_input = torch.cat(
            [matched_he, visual_ihc, matched_he_h.unsqueeze(1), visual_dab.unsqueeze(1)],
            dim=1,
        )
        pair_tokens = self.visual_proj(visual_input)

        query_tokens = self.query_q(self.query_proj(self.query_embeddings))
        visual_keys = self.visual_k(pair_tokens)
        query_scores = (query_tokens @ visual_keys.t()) / math.sqrt(self.hidden_dim)

        mil_scores = self.mil_attention_w(
            torch.tanh(self.mil_attention_v(pair_tokens))
            * torch.sigmoid(self.mil_attention_u(pair_tokens))
        ).squeeze(1)
        mil_weights = torch.softmax(mil_scores, dim=0)

        if self.disable_query_branch or self.disable_query_guidance:
            # A fair no-query baseline still learns which instances matter.
            # Repetition preserves the classifier input size and parameter count.
            query_weights = mil_weights.unsqueeze(0).expand(self.num_queries, -1)
        else:
            query_weights = torch.softmax(query_scores, dim=1)
        query_evidence = query_weights @ pair_tokens
        final_feature = query_evidence.reshape(-1)
        logit = self.classifier(final_feature.unsqueeze(0)).squeeze(0).squeeze(0)

        query_importance = torch.full(
            (self.num_queries,),
            1.0 / self.num_queries,
            device=pair_tokens.device,
        )
        outputs = {
            "logit": logit,
            "prob": torch.sigmoid(logit.detach()),
            "pair_tokens": pair_tokens,
            "final_feature": final_feature,
            "mil_weights": mil_weights,
            "guided_weights": query_weights.mean(dim=0),
            "query_weights": query_weights,
            "query_scores": query_scores,
            "query_component_scores": query_scores.unsqueeze(1),
            "query_evidence": query_evidence,
            "query_importance": query_importance,
            "aligned_dab": visual_dab,
            "alignment_evaluated_mask": getattr(self, "last_alignment_evaluated_mask", None) if s_align is not None else None,
            "align_indices": align_idx,
            "align_weights": align_weights,
            "align_mix_weights": align_mix,
            "s_feat": s_feat,
            "s_morph": s_morph,
            "s_base": s_base,
            "s_align": s_align,
            "pairwise_distance": pairwise_distance,
            "spatial_penalty": spatial_penalty,
        }
        return logit, outputs


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha_pos=0.5, alpha_neg=0.5):
        super().__init__()
        self.gamma = gamma
        self.alpha_pos = alpha_pos
        self.alpha_neg = alpha_neg

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.exp(-bce)
        alpha_t = targets * self.alpha_pos + (1.0 - targets) * self.alpha_neg
        loss = alpha_t * (1.0 - pt) ** self.gamma * bce
        return loss.mean()


def query_diversity_loss(query_weights):
    if query_weights.shape[0] <= 1:
        return torch.tensor(0.0, device=query_weights.device)
    w = F.normalize(query_weights, dim=1)
    sim = w @ w.t()
    eye = torch.eye(sim.shape[0], device=sim.device)
    off = sim * (1.0 - eye)
    return (off ** 2).sum() / max(sim.numel() - sim.shape[0], 1)


def attention_entropy(weights):
    eps = 1e-8
    entropy = -(weights * torch.log(weights + eps)).sum(dim=-1)
    denom = math.log(max(weights.shape[-1], 2))
    return (entropy / denom).mean()


def compute_metrics(labels, probs, n_bootstraps=2000, seed=42, groups=None):
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs).astype(float)
    preds = (probs >= 0.5).astype(int)
    if groups is None:
        groups = np.arange(len(labels))
    groups = np.asarray(groups)
    if len(groups) != len(labels):
        raise ValueError("groups must have the same length as labels")

    auc_value = float("nan")
    ci_low = float("nan")
    ci_high = float("nan")
    if len(np.unique(labels)) > 1:
        auc_value = roc_auc_score(labels, probs)
        rng = np.random.RandomState(seed)
        unique_groups = pd.unique(groups)
        group_members = [np.flatnonzero(groups == group) for group in unique_groups]
        aucs = []
        for _ in range(n_bootstraps):
            sampled_group_ids = rng.randint(
                0, len(unique_groups), size=len(unique_groups)
            )
            idx = np.concatenate([group_members[k] for k in sampled_group_ids])
            if len(np.unique(labels[idx])) < 2:
                continue
            aucs.append(roc_auc_score(labels[idx], probs[idx]))
        if aucs:
            ci_low, ci_high = np.percentile(aucs, [2.5, 97.5])

    return {
        "Accuracy": accuracy_score(labels, preds),
        "AUC": auc_value,
        "AUC_95_CI": f"({ci_low:.4f}, {ci_high:.4f})",
        "Precision": precision_score(labels, preds, zero_division=0),
        "Recall": recall_score(labels, preds, zero_division=0),
        "F1": f1_score(labels, preds, zero_division=0),
        "Confusion_Matrix": str(confusion_matrix(labels, preds).tolist()),
    }

def sigmoid_float(logit_value):
    logit_value = float(logit_value)
    if logit_value >= 0:
        z = math.exp(-min(logit_value, 80.0))
        return 1.0 / (1.0 + z)
    z = math.exp(min(logit_value, 80.0))
    return z / (1.0 + z)


def postprocess_probability_from_logit(logit_value, args):
    logit_value = float(logit_value)
    raw_prob = sigmoid_float(logit_value)
    mode = str(getattr(args, "prob_postprocess", "temperature")).lower()

    if mode == "temperature":
        temperature = max(float(getattr(args, "prob_temperature", 4.0)), 1e-6)
        scaled_logit = logit_value / temperature
        logit_clip = float(getattr(args, "prob_logit_clip", 8.0))
        if logit_clip > 0:
            scaled_logit = max(min(scaled_logit, logit_clip), -logit_clip)
        prob = sigmoid_float(scaled_logit)
    elif mode == "shrink":
        alpha = float(getattr(args, "prob_shrink_alpha", 0.9))
        alpha = max(0.0, min(alpha, 1.0))
        prob = 0.5 + alpha * (raw_prob - 0.5)
    elif mode == "clip":
        clip_min = float(getattr(args, "prob_clip_min", 1e-4))
        clip_max = float(getattr(args, "prob_clip_max", 1.0 - 1e-4))
        prob = min(max(raw_prob, clip_min), clip_max)
    else:
        prob = raw_prob

    return float(prob)


def save_queries(path):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query_text"])
        writer.writeheader()
        for query_id, query_text in DEFAULT_QUERIES:
            writer.writerow({"query_id": query_id, "query_text": query_text})


def save_split_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)


def detach_np(x):
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def save_case_intermediate(out_dir, split_name, sample, outputs):
    split_dir = ensure_dir(os.path.join(out_dir, "intermediates", split_name))
    path = os.path.join(split_dir, safe_name(sample["pair_case_name"]) + ".npz")

    data = {
        "case_id": np.array(sample["case_id"]),
        "project_name": np.array(sample["project_name"]),
        "pair_case_name": np.array(sample["pair_case_name"]),
        "he_file": np.array(sample["he_file"]),
        "ihc_file": np.array(sample["ihc_file"]),
        "label": detach_np(sample["label"]),
        "prob": detach_np(outputs["prob"]),
        "tile_names": np.asarray(sample["tile_names"]),
        "ihc_anchor_tile_names": np.asarray(sample["ihc_anchor_tile_names"]),
        "he_candidate_tile_names": np.asarray(sample["he_candidate_tile_names"]),
        "he_patch_paths": np.asarray(sample["he_patch_paths"]),
        "ihc_patch_paths": np.asarray(sample["ihc_patch_paths"]),
        "he_h": detach_np(sample["he_h"]),
        "ihc_h": detach_np(sample["ihc_h"]),
        "ihc_dab": detach_np(sample["ihc_dab"]),
        "aligned_dab": detach_np(outputs["aligned_dab"]),
        "mil_weights": detach_np(outputs["mil_weights"]),
        "guided_weights": detach_np(outputs["guided_weights"]),
        "query_weights": detach_np(outputs["query_weights"]),
        "query_scores": detach_np(outputs["query_scores"]),
        "query_component_scores": detach_np(outputs["query_component_scores"]),
        "query_evidence": detach_np(outputs["query_evidence"]),
        "query_importance": detach_np(outputs["query_importance"]),
        "align_indices": detach_np(outputs["align_indices"]),
        "align_weights": detach_np(outputs["align_weights"]),
        "align_mix_weights": detach_np(outputs["align_mix_weights"]),
        "s_feat": detach_np(outputs["s_feat"]),
        "s_morph": detach_np(outputs["s_morph"]),
        "s_base": detach_np(outputs["s_base"]),
        "alignment_evaluated_mask": detach_np(outputs.get("alignment_evaluated_mask")),
        "s_align": detach_np(outputs["s_align"]),
        "pairwise_distance": detach_np(outputs["pairwise_distance"]),
        "spatial_penalty": detach_np(outputs["spatial_penalty"]),
        "final_feature": detach_np(outputs["final_feature"]),
    }
    np.savez_compressed(path, **data)


def save_predictions_and_metrics(results_dir, split_name, rows, args):
    pred_df = pd.DataFrame(rows)
    if "predict_probability" not in pred_df.columns and "probability" in pred_df.columns:
        pred_df["predict_probability"] = pred_df["probability"]
    if "true_label" not in pred_df.columns and "label" in pred_df.columns:
        pred_df["true_label"] = pred_df["label"]

    should_postprocess = (
        str(getattr(args, "prob_postprocess", "temperature")).lower() != "none"
        and not str(split_name).endswith("_epoch_tmp")
    )
    if should_postprocess and "_raw_logit_for_storage" in pred_df.columns:
        pred_df["predict_probability"] = pred_df["_raw_logit_for_storage"].apply(
            lambda value: postprocess_probability_from_logit(value, args)
        )

    pred_df["predicted_label"] = (pred_df["predict_probability"] >= 0.5).astype(int)
    preferred_cols = [
        "project_name",
        "case_id",
        "ihc_file",
        "he_file",
        "pair_case_name",
        "true_label",
        "predicted_label",
        "predict_probability",
    ]
    hidden_cols = {"_raw_logit_for_storage"}
    other_cols = [col for col in pred_df.columns if col not in preferred_cols and col not in hidden_cols]
    pred_df = pred_df[[col for col in preferred_cols if col in pred_df.columns] + other_cols]
    pred_path = os.path.join(results_dir, f"{split_name}_wsi_predictions.csv")
    pred_df.to_csv(pred_path, index=False)

    metrics = compute_metrics(
        labels=pred_df["true_label"].values,
        probs=pred_df["predict_probability"].values,
        n_bootstraps=args.n_bootstraps,
        seed=args.seed,
        groups=(
            pred_df["case_id"].values
            if "case_id" in pred_df.columns
            else np.arange(len(pred_df))
        ),
    )
    metrics_df = pd.DataFrame([metrics])
    metrics_path = os.path.join(results_dir, f"{split_name}_wsi_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)

    print(f"\n--- {split_name} metrics ---")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    return metrics


def run_epoch(model, loader, optimizer, criterion, device, args, train=True, scaler=None):
    model.train(train)
    total_loss = 0.0
    rows = []
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
                logit, outputs = model(sample)
                loss = criterion(logit.view(1), label)

                if args.query_diversity_lambda > 0:
                    loss = loss + args.query_diversity_lambda * query_diversity_loss(outputs["query_weights"])
                if args.query_entropy_lambda > 0:
                    loss = loss + args.query_entropy_lambda * attention_entropy(outputs["query_weights"])
                if args.mil_entropy_lambda > 0:
                    loss = loss + args.mil_entropy_lambda * attention_entropy(outputs["guided_weights"])

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

        total_loss += float(loss.detach().cpu())
        raw_logit = float(logit.detach().cpu())
        prob = sigmoid_float(raw_logit)
        rows.append(
            {
                "project_name": sample["project_name"],
                "case_id": sample["case_id"],
                "ihc_file": sample["ihc_file"],
                "he_file": sample["he_file"],
                "pair_case_name": sample["pair_case_name"],
                "true_label": float(label.detach().cpu()),
                "predict_probability": prob,
                "_raw_logit_for_storage": raw_logit,
            }
        )
        iterator.set_postfix(loss=float(loss.detach().cpu()), prob=prob)

    return total_loss / max(len(loader), 1), rows


@torch.no_grad()
def evaluate(model, loader, device, args, split_name, save_intermediates=False):
    model.eval()
    rows = []
    amp_enabled = bool(args.amp and device.type == "cuda")
    for sample in tqdm(loader, desc=f"Evaluate {split_name}"):
        if sample is None:
            continue
        sample = to_device(sample, device)
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            logit, outputs = model(sample)
        raw_logit = float(logit.detach().cpu())
        prob = sigmoid_float(raw_logit)
        rows.append(
            {
                "project_name": sample["project_name"],
                "case_id": sample["case_id"],
                "ihc_file": sample["ihc_file"],
                "he_file": sample["he_file"],
                "pair_case_name": sample["pair_case_name"],
                "true_label": float(sample["label"].detach().cpu()),
                "predict_probability": prob,
                "_raw_logit_for_storage": raw_logit,
            }
        )
        if save_intermediates:
            save_case_intermediate(args.results_dir, split_name, sample, outputs)

    if not rows:
        return {}, rows
    metrics = save_predictions_and_metrics(args.results_dir, split_name, rows, args)
    return metrics, rows


def make_case_loader(df, args, split_name, training=False, shuffle=False):
    dataset = CaseMILDataset(df, args, split_name, training=training)
    dl_args = {
        "batch_size": 1,
        "num_workers": args.num_workers,
        "collate_fn": case_collate_fn,
        "pin_memory": bool(torch.cuda.is_available()),
    }
    if args.num_workers > 0:
        dl_args["persistent_workers"] = True
    return DataLoader(dataset, shuffle=shuffle, **dl_args)


def prebuild_case_caches(args, *dfs):
    rows = []
    for df in dfs:
        if df is not None and not df.empty:
            rows.append(df)
    if not rows:
        return

    all_df = pd.concat(rows, ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["project_name", "pair_case_name"], keep="first").reset_index(drop=True)
    cache_dir = get_cache_dir(args)

    print(f"[INFO] Prebuilding case tensor cache for {len(all_df)} unique pairs...")
    for _, row in tqdm(all_df.iterrows(), total=len(all_df), desc="Prebuild cache"):
        try:
            build_or_load_case_cache(row, args, cache_dir)
        except Exception as exc:
            print(f"[WARNING] Cache build failed for {row.get('pair_case_name', '')}: {exc}")
    print(f"[INFO] Cache prebuild finished: {cache_dir}")


def resolve_internal_split_paths(args):
    train_csv = args.train_split_csv
    test_csv = args.test_split_csv
    if not train_csv:
        candidate = split_csv_default_path(args.project_name, args.base_path, "train_split.csv")
        if os.path.exists(candidate):
            train_csv = candidate
    if not test_csv:
        candidate = split_csv_default_path(args.project_name, args.base_path, "test_split.csv")
        if os.path.exists(candidate):
            test_csv = candidate
    return train_csv, test_csv


def maybe_infer_image_feature_dim(args):
    feature_source = str(getattr(args, "feature_source", "fusion_tiles")).lower()
    if feature_source != "wsi_pt":
        return
    if not args.auto_feature_dim and int(args.feature_dim) > 0:
        return

    metadata_path = find_metadata_path(args.project_name, args.base_path)
    df = pd.read_csv(metadata_path)
    df = enrich_metadata_columns(df, args.project_name)

    for _, row in df.iterrows():
        try:
            he_path, ihc_path = get_wsi_feature_paths(
                row["pair_case_name"],
                args.project_name,
                args.image_encoder_name,
                args,
                he_file=str(row.get("he_file", "")),
                ihc_file=str(row.get("ihc_file", "")),
            )
            if os.path.exists(he_path):
                feat = load_wsi_feature_matrix(he_path)
                args.feature_dim = int(feat.shape[1])
                print(
                    f"[INFO] Auto inferred feature_dim={args.feature_dim} "
                    f"from {args.image_encoder_name}: {he_path}"
                )
                return
            if os.path.exists(ihc_path):
                feat = load_wsi_feature_matrix(ihc_path)
                args.feature_dim = int(feat.shape[1])
                print(
                    f"[INFO] Auto inferred feature_dim={args.feature_dim} "
                    f"from {args.image_encoder_name}: {ihc_path}"
                )
                return
        except Exception as exc:
            print(f"[WARNING] Feature-dim inference skipped one row: {exc}")

    raise RuntimeError(
        f"Could not infer feature_dim for image_encoder_name={args.image_encoder_name}. "
        "Please check *_features directories or pass --feature_dim manually."
    )


def build_dataframes(args):
    df = load_case_metadata(args.project_name, args.base_path, args=args)
    split_dir = ensure_dir(os.path.join(args.results_dir, "splits"))

    train_split_csv, test_split_csv = resolve_internal_split_paths(args)
    using_fixed_split = bool(train_split_csv and test_split_csv and os.path.exists(train_split_csv) and os.path.exists(test_split_csv))

    if args.train_on_all:
        train_df = df.copy()
        val_df = df.iloc[0:0].copy()
        test_df = df.iloc[0:0].copy()
    elif using_fixed_split:
        print(f"[INFO] Using fixed internal train split: {train_split_csv}")
        print(f"[INFO] Using fixed internal test split: {test_split_csv}")
        fixed_train_df = match_fixed_split(train_split_csv, df, args.results_dir, "internal_train_fixed")
        test_df = match_fixed_split(test_split_csv, df, args.results_dir, "internal_test_fixed")
        if args.val_size > 0 and len(fixed_train_df) >= 4:
            train_df, val_df, _ = stratified_split_cases(
                fixed_train_df,
                test_size=0,
                val_size=args.val_size,
                seed=args.seed,
            )
        else:
            train_df = fixed_train_df
            val_df = fixed_train_df.iloc[0:0].copy()
    else:
        print("[INFO] Fixed split CSVs were not found/provided; using random split.")
        train_df, val_df, test_df = stratified_split_cases(
            df,
            test_size=args.test_size,
            val_size=args.val_size,
            seed=args.seed,
        )

    external_df = pd.DataFrame()
    if args.external_project_name:
        external_df = load_case_metadata(
            args.external_project_name,
            args.base_path,
            expected_cases=args.expected_external_cases,
            args=args,
        )

    if args.prebuild_cache or args.cache_only:
        prebuild_case_caches(args, train_df, val_df, test_df, external_df)


    split_case_ids = {
        "train": set(train_df["case_id"].astype(str)),
        "val": set(val_df["case_id"].astype(str)),
        "internal_test": set(test_df["case_id"].astype(str)),
    }
    split_names = list(split_case_ids)
    for i, left_name in enumerate(split_names):
        for right_name in split_names[i + 1:]:
            overlap = split_case_ids[left_name] & split_case_ids[right_name]
            if overlap:
                raise ValueError(
                    f"Patient leakage between {left_name} and {right_name}: "
                    + ", ".join(sorted(overlap)[:10])
                )

    prepare_stain_scaler(args, train_df, build_or_load_case_cache, get_cache_dir(args))
    save_json(vars(args), os.path.join(args.results_dir, "config.json"))
    save_split_csv(train_df, os.path.join(split_dir, "train_cases.csv"))
    save_split_csv(val_df, os.path.join(split_dir, "val_cases.csv"))
    save_split_csv(test_df, os.path.join(split_dir, "internal_test_cases.csv"))
    if not external_df.empty:
        save_split_csv(external_df, os.path.join(split_dir, f"external_{args.external_project_name}_cases.csv"))

    print(
        f"[INFO] Split cases: train={len(train_df)}, val={len(val_df)}, "
        f"internal_test={len(test_df)}, external={len(external_df)}"
    )
    if args.cache_only:
        print("[DONE] Cache construction and training-only stain scaler fitting completed.")
        raise SystemExit(0)
    return train_df, val_df, test_df, external_df


def build_loaders(args):
    train_df, val_df, test_df, external_df = build_dataframes(args)
    train_loader = make_case_loader(train_df, args, "train", training=True, shuffle=True)
    val_loader = make_case_loader(val_df, args, "val", training=False, shuffle=False)
    test_loader = make_case_loader(test_df, args, "internal_test", training=False, shuffle=False)
    external_loader = make_case_loader(external_df, args, "external", training=False, shuffle=False) if not external_df.empty else None
    return train_df, val_df, test_df, external_df, train_loader, val_loader, test_loader, external_loader


def compute_pos_weight(train_df, device):
    labels = train_df["label"].astype(int).values
    pos = max(int(labels.sum()), 1)
    neg = max(int(len(labels) - labels.sum()), 1)
    return torch.tensor([neg / pos], dtype=torch.float32, device=device)


def make_criterion(args, train_df, device):
    if args.loss == "focal":
        labels = train_df["label"].astype(int).values
        pos_ratio = float(labels.mean()) if len(labels) else 0.5
        alpha_pos = 1.0 - pos_ratio if 0.0 < pos_ratio < 1.0 else 0.5
        alpha_neg = pos_ratio if 0.0 < pos_ratio < 1.0 else 0.5
        return FocalLoss(
            gamma=args.focal_gamma,
            alpha_pos=alpha_pos,
            alpha_neg=alpha_neg,
        ).to(device)
    return nn.BCEWithLogitsLoss(pos_weight=compute_pos_weight(train_df, device)).to(device)


def get_model_selection_loader(val_loader, val_df):
    if len(val_df) == 0:
        raise ValueError(
            "A non-empty validation split is required for checkpoint selection. "
            "Set --val_size to a value greater than zero. Test cohorts are never "
            "used for early stopping or model selection."
        )
    return "val", val_loader, len(val_df)


def train(args):
    set_seed(args.seed)
    ensure_dir(args.results_dir)
    maybe_infer_image_feature_dim(args)
    save_json(vars(args), os.path.join(args.results_dir, "config.json"))
    save_queries(os.path.join(args.results_dir, "queries.csv"))

    device = torch.device(args.device if torch.cuda.is_available() or "cpu" in args.device else "cpu")
    print(f"[INFO] Device: {device}")
    if args.tf32 and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
        print("[INFO] TF32 enabled for CUDA matmul/cudnn.")

    train_df, val_df, test_df, external_df, train_loader, val_loader, test_loader, external_loader = build_loaders(args)

    query_embeddings, query_embedding_metadata = load_query_embeddings(args, device)
    np.save(
        os.path.join(args.results_dir, "query_embeddings.npy"),
        query_embeddings.detach().cpu().numpy(),
    )
    save_json(
        query_embedding_metadata,
        os.path.join(args.results_dir, "query_embedding_metadata.json"),
    )

    model = VGMS(
        query_embeddings=query_embeddings,
        feature_dim=args.feature_dim,
        morph_dim=5,
        hidden_dim=args.hidden_dim,
        align_dim=args.align_dim,
        dropout=args.dropout,
        align_topk=args.align_topk,
        temperature=args.align_temperature,
        disable_morph_alignment=args.disable_morph_alignment,
        disable_query_guidance=args.disable_query_guidance,
        disable_query_branch=args.disable_query_branch,
        disable_alignment=args.disable_alignment,
        input_mode=args.input_mode,
        alignment_mode=args.alignment_mode,
        alignment_lambda=args.alignment_lambda,
        spatial_gamma=args.spatial_gamma,
        spatial_sigma=args.spatial_sigma,
        patch_stride=args.patch_size,
    ).to(device)

    criterion = make_criterion(args, train_df, device)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.min_lr,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=bool(args.amp and device.type == "cuda"))

    best_score = -float("inf")
    best_epoch = 0
    patience_counter = 0
    best_path = os.path.join(args.results_dir, "best_model.pth")
    selection_split_name, selection_loader, selection_size = get_model_selection_loader(
        val_loader, val_df
    )
    print(
        f"[INFO] Model selection and early stopping use "
        f"{selection_split_name} AUC (n={selection_size})."
    )

    for epoch in range(1, args.epochs + 1):
        print("=" * 100)
        print(f"[INFO] Epoch {epoch}/{args.epochs}")
        train_loss, train_rows = run_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            args,
            train=True,
            scaler=scaler,
        )
        scheduler.step()

        train_metrics = compute_metrics(
            labels=[row["true_label"] for row in train_rows],
            probs=[row["predict_probability"] for row in train_rows],
            n_bootstraps=100,
            seed=args.seed,
        )
        print(f"[INFO] Train loss={train_loss:.4f}, Train AUC={train_metrics['AUC']}")

        selection_metrics, _ = evaluate(
            model,
            selection_loader,
            device,
            args,
            f"{selection_split_name}_epoch_tmp",
            save_intermediates=False,
        )
        selection_auc = selection_metrics.get("AUC", float("nan"))
        score = selection_auc if not np.isnan(selection_auc) else selection_metrics.get("Accuracy", 0.0)
        print(
            f"[INFO] Model-selection split={selection_split_name}, "
            f"AUC={selection_auc}, score={score:.4f}"
        )

        if score > best_score:
            best_score = score
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": vars(args),
                    "query_defs": DEFAULT_QUERIES,
                    "query_embeddings": query_embeddings.detach().cpu(),
                },
                best_path,
            )
            print(f"[INFO] New best score={best_score:.4f}; saved {best_path}")
        else:
            patience_counter += 1
            print(f"[INFO] No improvement. patience={patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            print(f"[INFO] Early stopping at epoch {epoch}")
            break

    print("=" * 100)
    print(f"[INFO] Loading best model from epoch {best_epoch}: {best_path}")
    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model"])

    evaluate(model, train_loader, device, args, "train_best", save_intermediates=args.save_train_intermediates)
    if len(val_df) > 0:
        evaluate(model, val_loader, device, args, "val_best", save_intermediates=True)
    if not args.skip_test_evaluation and len(test_df) > 0:
        evaluate(model, test_loader, device, args, "internal_test_best", save_intermediates=True)

    if not args.skip_test_evaluation and external_loader is not None and len(external_df) > 0:
        evaluate(
            model,
            external_loader,
            device,
            args,
            f"external_{args.external_project_name}",
            save_intermediates=True,
        )

    if not args.skip_test_evaluation:
        write_single_experiment_summary(
            args.results_dir,
            experiment_name=os.path.basename(os.path.abspath(args.results_dir)) or "main_experiment",
            prefix="summary",
        )

    print(f"[DONE] Results saved to: {args.results_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train distance-aware Top-64 H&E-IHC VGMS without clinical variables."
    )
    parser.add_argument("--project_name", type=str, default="cohort1")
    parser.add_argument("--external_project_name", type=str, default="cohort2")
    parser.add_argument("--base_path", type=str, default="../data/patches")
    parser.add_argument("--results_dir", type=str, default="./results/vgms_main")
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="Optional shared case tensor cache directory. Use this to reuse cache across ablation result dirs.",
    )
    parser.add_argument(
        "--train_split_csv",
        type=str,
        default="",
        help="Fixed internal train split CSV. If empty, auto-detect train_split.csv beside pair_labels_with_case_id.csv.",
    )
    parser.add_argument(
        "--test_split_csv",
        type=str,
        default="",
        help="Fixed internal test split CSV. If empty, auto-detect test_split.csv beside pair_labels_with_case_id.csv.",
    )
    parser.add_argument(
        "--expected_external_cases",
        type=int,
        default=101,
        help="Warn if external project case count after dedupe differs from this. Set 0 to disable.",
    )

    parser.add_argument("--feature_dim", type=int, default=768)
    parser.add_argument(
        "--feature_source",
        type=str,
        default="fusion_tiles",
        choices=["fusion_tiles", "wsi_pt"],
        help="fusion_tiles reads one paired HE/IHC patch feature pt per tile; wsi_pt reads one WSI pt per HE/IHC slide and splits it back to patches by old tile order.",
    )
    parser.add_argument(
        "--image_encoder_name",
        type=str,
        default="CHIEF",
        help="Base image encoder name used when --feature_source wsi_pt, e.g. TITAN, UNI, Virchow2.",
    )
    parser.set_defaults(auto_feature_dim=True)
    parser.add_argument(
        "--auto_feature_dim",
        action="store_true",
        help="Automatically infer WSI feature dimension. Enabled by default.",
    )
    parser.add_argument(
        "--no_auto_feature_dim",
        dest="auto_feature_dim",
        action="store_false",
        help="Disable automatic feature_dim inference for WSI pt features.",
    )
    parser.add_argument("--allow_wsi_feature_truncate", action="store_true", default=True)
    parser.add_argument("--no_allow_wsi_feature_truncate", dest="allow_wsi_feature_truncate", action="store_false")
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--align_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.35)
    parser.add_argument(
        "--input_mode",
        type=str,
        default="multistain",
        choices=["he", "ihc", "multistain"],
        help="Ablation input mode: only HE, only IHC, or aligned multi-stain.",
    )
    parser.add_argument(
        "--alignment_mode",
        type=str,
        default="both",
        choices=["none", "feature", "morph", "both"],
        help="Multi-stain alignment ablation: no alignment, feature-only, morphology-only, or fixed validation-selected combination.",
    )
    parser.add_argument(
        "--align_topk",
        type=int,
        default=64,
        help="Number of H&E candidates softly aggregated for each IHC anchor.",
    )
    parser.add_argument("--align_temperature", type=float, default=1.0)
    parser.add_argument(
        "--spatial_gamma",
        type=float,
        default=0.25,
        help="Weight of the Gaussian spatial distance penalty in Eq. (4).",
    )
    parser.add_argument(
        "--spatial_sigma",
        type=float,
        default=8.0,
        help="Gaussian spatial scale in patch-grid units.",
    )
    parser.add_argument(
        "--alignment_lambda",
        type=float,
        default=0.5,
        help="Fixed morphology weight lambda; feature weight is 1-lambda. Select on validation data only.",
    )
    parser.add_argument("--max_patches_per_case", type=int, default=1024)
    parser.add_argument("--patch_size", type=int, default=250)
    parser.add_argument("--ssim_map_size", type=int, default=32)

    parser.add_argument(
        "--text_encoder_type",
        type=str,
        default="clinicalbert",
        choices=["clinicalbert", "tfidf", "clip", "qwen", "qwen2.5-0.5b", "qwen2_5_0_5b"],
        help="Query text encoder for ablation.",
    )
    parser.add_argument("--text_model", type=str, default="emilyalsentzer/Bio_ClinicalBERT")
    parser.add_argument("--clip_text_model", type=str, default="openai/clip-vit-base-patch32")
    parser.add_argument("--qwen_text_model", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--text_max_length", type=int, default=64)
    parser.add_argument(
        "--text_encoder_dtype",
        type=str,
        default="auto",
        choices=["auto", "float32", "float16", "bfloat16"],
        help="Dtype for CLIP/Qwen query encoding. auto uses fp16 on CUDA and fp32 on CPU.",
    )
    parser.add_argument(
        "--qwen_pooling",
        type=str,
        default="last_token",
        choices=["last_token", "mean"],
        help="Pooling strategy for decoder-only Qwen query embeddings.",
    )
    parser.add_argument("--tfidf_ngram_max", type=int, default=2)
    parser.add_argument("--query_embedding_path", type=str, default="")
    parser.add_argument(
        "--query_encoder_device",
        type=str,
        default="cpu",
        help="Device for one-time neural query encoding. CPU saves GPU memory.",
    )
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--morphology_mode",
        type=str,
        default="macenko",
        choices=["fast_gray", "macenko", "none"],
        help="fast_gray is much faster; macenko matches the stain-separation idea but is CPU-heavy.",
    )

    parser.add_argument("--stain_scalar_mode", choices=["train_standardized", "legacy_patch_mean"],
                        default="train_standardized",
                        help="Fusion scalars: raw concentration means standardized using training-only channel statistics, or patch-relative means.")

    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--loss", type=str, default="focal", choices=["bce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--grad_clip", type=float, default=5.0)

    parser.add_argument("--query_diversity_lambda", type=float, default=0.0)
    parser.add_argument("--query_entropy_lambda", type=float, default=0.0)
    parser.add_argument("--mil_entropy_lambda", type=float, default=0.0)

    parser.add_argument("--test_size", type=float, default=0.3)
    parser.add_argument(
        "--val_size",
        type=float,
        default=2.0 / 7.0,
        help="Validation fraction within the 70% development pool; 2/7 yields the paper's 5:2:3 split.",
    )
    parser.add_argument("--train_on_all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_bootstraps", type=int, default=2000)

    parser.add_argument(
        "--prob_postprocess",
        type=str,
        default="temperature",
        choices=["none", "temperature", "shrink", "clip"],
        help="Post-process saved final prediction probabilities only; the paper protocol uses temperature scaling with T=4. Training and model selection stay unchanged.",
    )
    parser.add_argument(
        "--prob_temperature",
        type=float,
        default=4.0,
        help="Saved probability uses sigmoid(raw_logit / T). Larger T is less extreme.",
    )
    parser.add_argument(
        "--prob_logit_clip",
        type=float,
        default=0.0,
        help="Optional absolute logit clip after temperature scaling. Default <=0 disables clipping and preserves ranking.",
    )
    parser.add_argument(
        "--prob_shrink_alpha",
        type=float,
        default=0.9,
        help="Shrink mode: p'=0.5+alpha*(p-0.5). Smaller alpha is less extreme.",
    )
    parser.add_argument("--prob_clip_min", type=float, default=1e-4)
    parser.add_argument("--prob_clip_max", type=float, default=1.0 - 1e-4)

    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--force_rebuild_cache", action="store_true")
    parser.add_argument(
        "--allow_missing_morphology",
        action="store_true",
        help="Allow missing patch images by substituting zero morphology maps; not paper-faithful.",
    )
    parser.add_argument("--prebuild_cache", action="store_true")
    parser.add_argument("--cache_only", action="store_true")
    parser.add_argument(
        "--skip_test_evaluation",
        action="store_true",
        help="Train/select on train-validation data without evaluating internal or external tests.",
    )
    parser.set_defaults(save_train_intermediates=True)
    parser.add_argument(
        "--save_train_intermediates",
        action="store_true",
        help="Save train split intermediate npz files. Enabled by default.",
    )
    parser.add_argument(
        "--no_save_train_intermediates",
        dest="save_train_intermediates",
        action="store_false",
        help="Disable saving train split intermediate npz files.",
    )
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--tf32", action="store_true")

    parser.add_argument("--disable_morph_alignment", action="store_true")
    parser.add_argument("--disable_query_guidance", action="store_true")
    parser.add_argument("--disable_query_branch", action="store_true")
    parser.add_argument("--disable_alignment", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
