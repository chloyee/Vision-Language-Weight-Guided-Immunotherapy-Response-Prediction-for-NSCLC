"""
Query-guided WSI heatmap generation for de-identified H&E-IHC pairs.

Example:
    python query_attention_heatmaps_chief_ctranspath.py \
        --slides_csv ./examples/slides.csv \
        --ctranspath_weight ./model_weight/CHIEF_CTransPath.pth \
        --out_dir ./query_attention_heatmap_outputs \
        --tile_size 80 --batch_size 64 --amp

Important:
    ClinicalBERT and CTransPath are not trained as a paired vision-language
    model. Without --alignment_ckpt, this script uses an identity mapping
    because both features are usually 768-D. It can generate exploratory
    heatmaps, but task-faithful query attention should use a trained
    text-to-image projection or cross-attention checkpoint.
"""

import os
import csv
import math
import argparse
import warnings
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
import openslide
from openslide import open_slide, ImageSlide
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from torchvision import transforms


Image.MAX_IMAGE_PIXELS = None
warnings.filterwarnings("ignore")


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


@dataclass
class SlideSpec:
    pair_id: str
    label: int
    modality: str
    path: str
    stain_type: str


def load_slide_specs(csv_path):
    required = {"pair_id", "label", "modality", "path", "stain_type"}
    specs = []
    with open(csv_path, "r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Slide manifest is missing columns: {sorted(missing)}")
        for row in reader:
            specs.append(
                SlideSpec(
                    pair_id=str(row["pair_id"]),
                    label=int(row["label"]),
                    modality=str(row["modality"]).upper(),
                    path=str(row["path"]),
                    stain_type=str(row["stain_type"]).upper(),
                )
            )
    if not specs:
        raise ValueError(f"No slides were listed in {csv_path}")
    return specs


def safe_name(text, max_len=80):
    out = []
    for char in text.lower():
        if char.isalnum():
            out.append(char)
        elif char in [" ", "-", "_", "/", "."]:
            out.append("_")
    name = "".join(out)
    while "__" in name:
        name = name.replace("__", "_")
    name = name.strip("_")
    return name[:max_len] if name else "item"


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def open_wsi_slide(wsi_path):
    """
    Open WSI files robustly across openslide-python versions.

    Some environments do not provide openslide.detect_format. The original
    visualization script only used it for logging, so here we keep it optional
    and always try open_slide() first.
    """
    if not os.path.exists(wsi_path):
        raise FileNotFoundError(f"WSI not found: {wsi_path}")

    try:
        if hasattr(openslide, "detect_format"):
            fmt = openslide.detect_format(wsi_path)
            print(f"[INFO] openslide.detect_format: {fmt}")
        else:
            print("[WARNING] openslide.detect_format is not available; skip format check.")
    except Exception as exc:
        print(f"[WARNING] openslide.detect_format failed: {exc}")

    open_slide_error = None
    try:
        slide = open_slide(wsi_path)
        print(f"[INFO] Opened slide by open_slide(): {type(slide)}")
        print(f"[INFO] Dimensions: {slide.dimensions}")
        return slide
    except Exception as exc:
        open_slide_error = exc
        print(f"[WARNING] open_slide() failed, trying PIL ImageSlide fallback: {exc}")

    try:
        pil_img = Image.open(wsi_path)
        slide = ImageSlide(pil_img)
        print(f"[INFO] Opened slide by ImageSlide(PIL): {type(slide)}")
        print(f"[INFO] Dimensions: {slide.dimensions}")
        return slide
    except Exception as pil_error:
        raise RuntimeError(
            f"Failed to open slide: {wsi_path}\n"
            f"open_slide() error: {repr(open_slide_error)}\n"
            f"ImageSlide(PIL) error: {repr(pil_error)}"
        )


def compute_tissue_score(pil_img):
    rgb = np.asarray(pil_img.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    white_ratio = float(np.mean(gray > 220))
    saturation_score = float(np.mean(hsv[:, :, 1] > 20))
    edges = cv2.Canny(gray, 50, 150)
    edge_score = float(np.mean(edges > 0))

    return 0.5 * (1.0 - white_ratio) + 0.3 * saturation_score + 0.2 * edge_score


def extract_dab_rgb_from_patch(pil_img):
    rgb = np.asarray(pil_img.convert("RGB")).astype(np.float32)
    od = -np.log((rgb + 1.0) / 255.0)

    # Ruifrok-Johnston H-DAB approximate stain matrix.
    stain_matrix = np.array(
        [
            [0.650, 0.704, 0.286],
            [0.268, 0.570, 0.776],
            [0.711, 0.423, 0.561],
        ],
        dtype=np.float32,
    )
    stain_matrix = stain_matrix / np.linalg.norm(stain_matrix, axis=1, keepdims=True)
    concentrations = od.reshape(-1, 3) @ np.linalg.pinv(stain_matrix).T
    dab = concentrations[:, 1].reshape(rgb.shape[:2])
    dab = dab - float(dab.min())
    if float(dab.max()) > 0:
        dab = dab / float(dab.max())
    dab_img = (255.0 * (1.0 - dab)).clip(0, 255).astype(np.uint8)
    dab_rgb = np.stack([dab_img, dab_img, dab_img], axis=-1)
    return Image.fromarray(dab_rgb)


def extract_patches_from_wsi(
    wsi_path,
    out_patch_dir,
    tile_size,
    stride,
    tissue_threshold,
    stain_type,
    ihc_use_dab,
    max_patches,
):
    ensure_dir(out_patch_dir)
    slide = open_wsi_slide(wsi_path)
    w, h = slide.dimensions

    n_cols = max(1, math.floor((w - tile_size) / stride) + 1)
    n_rows = max(1, math.floor((h - tile_size) / stride) + 1)

    print(f"[INFO] Patch grid: {n_cols} cols x {n_rows} rows")
    print(f"[INFO] tile_size={tile_size}, stride={stride}, tissue_threshold={tissue_threshold}")

    coords = []
    stop = False
    for row in range(n_rows):
        if stop:
            break
        for col in range(n_cols):
            x = col * stride
            y = row * stride

            if x + tile_size > w or y + tile_size > h:
                continue

            patch = slide.read_region(
                location=(x, y),
                level=0,
                size=(tile_size, tile_size),
            ).convert("RGB")

            score = compute_tissue_score(patch)
            if score < tissue_threshold:
                continue

            processed = patch
            if stain_type.upper() == "IHC" and ihc_use_dab:
                processed = extract_dab_rgb_from_patch(processed)

            patch_name = f"{x}_{y}.png"
            processed.save(os.path.join(out_patch_dir, patch_name))

            coords.append(
                {
                    "patch_name": patch_name,
                    "x": x,
                    "y": y,
                    "col": col,
                    "row": row,
                    "tissue_score": score,
                }
            )

            if max_patches > 0 and len(coords) >= max_patches:
                stop = True
                break

    slide.close()
    print(f"[INFO] Extracted tissue patches: {len(coords)}")
    return coords, w, h


def save_patch_coords(coords, csv_path):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["patch_name", "x", "y", "col", "row", "tissue_score"],
        )
        writer.writeheader()
        for item in coords:
            writer.writerow(item)


def load_patch_coords(csv_path):
    coords = []
    with open(csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            coords.append(
                {
                    "patch_name": row["patch_name"],
                    "x": int(float(row["x"])),
                    "y": int(float(row["y"])),
                    "col": int(float(row["col"])),
                    "row": int(float(row["row"])),
                    "tissue_score": float(row["tissue_score"]),
                }
            )
    return coords


def build_patch_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )


def load_ctranspath_encoder(weight_path, device):
    try:
        from CHIEF.models.ctran import ctranspath
    except Exception as exc:
        raise ImportError(
            "Could not import CHIEF.models.ctran. Run this script from the "
            "WSI-HGNN-main project root, or make sure the CHIEF package is on PYTHONPATH."
        ) from exc

    if not os.path.exists(weight_path):
        raise FileNotFoundError(
            f"CTransPath checkpoint not found: {weight_path}\n"
            "Expected CHIEF_CTransPath.pth. Pass --ctranspath_weight if it is elsewhere."
        )

    model = ctranspath()
    if hasattr(model, "head"):
        model.head = nn.Identity()

    checkpoint = torch.load(weight_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))

    cleaned = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ["module.", "model.", "encoder."]:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        cleaned[new_key] = value

    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARNING] Missing CTransPath keys: {len(missing)}")
        print(f"[WARNING] First missing keys: {missing[:5]}")
    if unexpected:
        print(f"[WARNING] Unexpected CTransPath keys: {len(unexpected)}")
        print(f"[WARNING] First unexpected keys: {unexpected[:5]}")

    model.to(device)
    model.eval()
    return model


def load_text_encoder(text_model_name, device, local_files_only, max_length):
    try:
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:
        raise ImportError(
            "transformers is required for ClinicalBERT text query encoding."
        ) from exc

    print(f"[INFO] Loading text encoder: {text_model_name}")
    tokenizer = AutoTokenizer.from_pretrained(
        text_model_name,
        local_files_only=local_files_only,
    )
    model = AutoModel.from_pretrained(
        text_model_name,
        local_files_only=local_files_only,
    )
    model.to(device)
    model.eval()
    return tokenizer, model, max_length


@torch.inference_mode()
def encode_queries(query_defs, tokenizer, text_model, max_length, device):
    texts = [item[1] for item in query_defs]
    batch = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch = {key: value.to(device) for key, value in batch.items()}
    output = text_model(**batch)

    if hasattr(output, "pooler_output") and output.pooler_output is not None:
        embeddings = output.pooler_output
    else:
        embeddings = output.last_hidden_state[:, 0]

    return embeddings.detach().float().cpu()


def normalize_encoder_output(output):
    if isinstance(output, (tuple, list)):
        output = output[0]
    if isinstance(output, dict):
        output = output.get("features", output.get("last_hidden_state"))

    if output.ndim == 4:
        output = F.adaptive_avg_pool2d(output, (1, 1)).flatten(1)
    elif output.ndim == 3:
        output = output[:, 0] if output.shape[1] > 1 else output.squeeze(1)
    elif output.ndim > 2:
        output = output.flatten(1)

    return output


@torch.inference_mode()
def encode_patches(
    patch_dir,
    coords,
    encoder,
    device,
    batch_size,
    use_amp,
):
    transform = build_patch_transform()
    features = []
    valid_coords = []

    for start in range(0, len(coords), batch_size):
        batch_coords = coords[start:start + batch_size]
        images = []
        kept = []

        for item in batch_coords:
            patch_path = os.path.join(patch_dir, item["patch_name"])
            if not os.path.exists(patch_path):
                print(f"[WARNING] Missing patch image, skipped: {patch_path}")
                continue
            try:
                img = Image.open(patch_path).convert("RGB")
                images.append(transform(img))
                kept.append(item)
            except Exception as exc:
                print(f"[WARNING] Failed to read patch {patch_path}: {exc}")

        if not images:
            continue

        batch = torch.stack(images, dim=0).to(device)
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=bool(use_amp and device.type == "cuda"),
        ):
            output = encoder(batch)
            output = normalize_encoder_output(output)

        features.append(output.detach().float().cpu())
        valid_coords.extend(kept)

        done = min(start + batch_size, len(coords))
        print(f"[INFO] Encoded patches: {done}/{len(coords)}", end="\r")

    print("")
    if not features:
        raise RuntimeError("No patch features were produced.")

    return torch.cat(features, dim=0), valid_coords


def build_query_projection(text_dim, image_dim, alignment_ckpt, device):
    projection = nn.Linear(text_dim, image_dim, bias=False)

    if text_dim == image_dim:
        with torch.no_grad():
            projection.weight.copy_(torch.eye(image_dim, text_dim))
    else:
        nn.init.xavier_uniform_(projection.weight)

    if alignment_ckpt:
        if not os.path.exists(alignment_ckpt):
            raise FileNotFoundError(f"Alignment checkpoint not found: {alignment_ckpt}")
        checkpoint = torch.load(alignment_ckpt, map_location="cpu")
        state = checkpoint.get("projection", checkpoint.get("state_dict", checkpoint))

        if "weight" in state:
            projection.load_state_dict(state, strict=True)
        else:
            cleaned = {}
            for key, value in state.items():
                if key.startswith("text_to_image."):
                    cleaned[key.replace("text_to_image.", "")] = value
                elif key.startswith("projection."):
                    cleaned[key.replace("projection.", "")] = value
            projection.load_state_dict(cleaned, strict=True)
        print(f"[INFO] Loaded trained text-to-image alignment: {alignment_ckpt}")
    else:
        if text_dim != image_dim:
            raise RuntimeError(
                "Text and image feature dimensions differ and no --alignment_ckpt was provided. "
                "Train or provide a text-to-image projection checkpoint first."
            )
        print(
            "[WARNING] No --alignment_ckpt provided. Using identity text-to-image mapping. "
            "This runs without training, but ClinicalBERT and CTransPath are not truly aligned."
        )

    projection.to(device)
    projection.eval()
    return projection


@torch.inference_mode()
def compute_query_patch_attention(
    patch_features,
    query_embeddings,
    projection,
    device,
    temperature,
):
    image_features = F.normalize(patch_features.to(device), dim=1)
    query_features = projection(query_embeddings.to(device))
    query_features = F.normalize(query_features, dim=1)

    raw_scores = image_features @ query_features.t()
    raw_scores = raw_scores / max(float(temperature), 1e-6)
    attention = torch.softmax(raw_scores.t(), dim=1)

    return attention.cpu(), raw_scores.t().cpu()


def build_patch_value_map(coords, values, thumb_w, thumb_h, tile_size, downsample):
    value_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    count_map = np.zeros((thumb_h, thumb_w), dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)

    for item, value in zip(coords, values):
        x = int(item["x"])
        y = int(item["y"])

        x0 = int(x / downsample)
        y0 = int(y / downsample)
        x1 = int((x + tile_size) / downsample)
        y1 = int((y + tile_size) / downsample)

        x0 = max(0, min(x0, thumb_w - 1))
        x1 = max(0, min(x1, thumb_w))
        y0 = max(0, min(y0, thumb_h - 1))
        y1 = max(0, min(y1, thumb_h))

        if x1 <= x0 or y1 <= y0:
            continue

        value_map[y0:y1, x0:x1] += float(value)
        count_map[y0:y1, x0:x1] += 1.0

    mask = count_map > 0
    if np.any(mask):
        value_map[mask] = value_map[mask] / count_map[mask]

    return value_map, mask


def masked_gaussian_smooth(value_map, mask, sigma):
    if sigma <= 0 or not np.any(mask):
        return value_map

    value = np.where(mask, value_map, 0).astype(np.float32)
    weight = mask.astype(np.float32)
    blurred_value = cv2.GaussianBlur(value, (0, 0), sigmaX=sigma, sigmaY=sigma)
    blurred_weight = cv2.GaussianBlur(weight, (0, 0), sigmaX=sigma, sigmaY=sigma)

    out = np.zeros_like(value_map, dtype=np.float32)
    valid = blurred_weight > 1e-6
    out[valid] = blurred_value[valid] / blurred_weight[valid]
    out[~mask] = 0
    return out


def normalize_on_mask(value_map, mask, low_percentile=2.0, high_percentile=98.0):
    out = np.zeros_like(value_map, dtype=np.float32)
    if not np.any(mask):
        return out

    vals = value_map[mask]
    vmin = float(np.percentile(vals, low_percentile))
    vmax = float(np.percentile(vals, high_percentile))
    if vmax <= vmin:
        vmax = float(vals.max())
        vmin = float(vals.min())

    if vmax > vmin:
        clipped = np.clip(value_map, vmin, vmax)
        out[mask] = (clipped[mask] - vmin) / (vmax - vmin + 1e-8)
    return out


def render_attention_map(
    thumbnail_np,
    heatmap,
    mask,
    overlay_path,
    pure_path,
    cmap_name,
    alpha,
):
    cmap = cm.get_cmap(cmap_name)
    heat_rgb = (cmap(np.clip(heatmap, 0, 1))[:, :, :3] * 255).astype(np.uint8)

    overlay = thumbnail_np.copy()
    blended = (
        (1.0 - alpha) * thumbnail_np.astype(np.float32)
        + alpha * heat_rgb.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)
    overlay[mask] = blended[mask]

    pure = np.zeros_like(thumbnail_np, dtype=np.uint8)
    pure[mask] = heat_rgb[mask]

    Image.fromarray(overlay).save(overlay_path)
    Image.fromarray(pure).save(pure_path)


def render_matplotlib_style(
    thumbnail_np,
    heatmap,
    mask,
    save_path,
    title,
    cmap_name,
    alpha,
):
    masked_heatmap = np.ma.masked_where(~mask, heatmap)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(thumbnail_np)
    im = ax.imshow(
        masked_heatmap,
        cmap=cmap_name,
        alpha=alpha,
        vmin=0,
        vmax=1,
        interpolation="spline16",
    )
    ax.set_title(title, fontsize=12)
    ax.axis("off")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized query attention", fontsize=10)
    fig.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=180)
    plt.close(fig)


def draw_query_heatmaps(
    wsi_path,
    coords,
    attention_values,
    raw_values,
    query_id,
    query_text,
    out_dir,
    tile_size,
    downsample,
    cmap_name,
    alpha,
    smooth_sigma,
):
    ensure_dir(out_dir)
    slide = open_wsi_slide(wsi_path)
    w, h = slide.dimensions
    thumb_w = max(1, int(w // downsample))
    thumb_h = max(1, int(h // downsample))

    thumbnail = slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB")
    if thumbnail.size != (thumb_w, thumb_h):
        thumbnail = thumbnail.resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
    thumbnail_np = np.asarray(thumbnail)

    attn_map, mask = build_patch_value_map(
        coords=coords,
        values=attention_values,
        thumb_w=thumb_w,
        thumb_h=thumb_h,
        tile_size=tile_size,
        downsample=downsample,
    )
    raw_map, raw_mask = build_patch_value_map(
        coords=coords,
        values=raw_values,
        thumb_w=thumb_w,
        thumb_h=thumb_h,
        tile_size=tile_size,
        downsample=downsample,
    )
    mask = np.logical_or(mask, raw_mask)

    if smooth_sigma > 0:
        attn_map = masked_gaussian_smooth(attn_map, mask, smooth_sigma)
        raw_map = masked_gaussian_smooth(raw_map, mask, smooth_sigma)

    attn_vis = normalize_on_mask(attn_map, mask)
    raw_vis = normalize_on_mask(raw_map, mask)

    query_slug = safe_name(query_id)
    prefix = os.path.join(out_dir, query_slug)

    np.save(prefix + "_attention_weight_map.npy", attn_map)
    np.save(prefix + "_raw_score_map.npy", raw_map)
    np.save(prefix + "_mask.npy", mask.astype(np.uint8))

    attn_overlay = prefix + "_attention_weight_overlay.png"
    attn_pure = prefix + "_attention_weight_pure.png"
    raw_overlay = prefix + "_raw_score_overlay.png"
    raw_pure = prefix + "_raw_score_pure.png"
    attn_matplotlib = prefix + "_attention_weight_matplotlib.png"
    raw_matplotlib = prefix + "_raw_score_matplotlib.png"

    render_attention_map(
        thumbnail_np=thumbnail_np,
        heatmap=attn_vis,
        mask=mask,
        overlay_path=attn_overlay,
        pure_path=attn_pure,
        cmap_name=cmap_name,
        alpha=alpha,
    )
    render_attention_map(
        thumbnail_np=thumbnail_np,
        heatmap=raw_vis,
        mask=mask,
        overlay_path=raw_overlay,
        pure_path=raw_pure,
        cmap_name=cmap_name,
        alpha=alpha,
    )
    render_matplotlib_style(
        thumbnail_np=thumbnail_np,
        heatmap=attn_vis,
        mask=mask,
        save_path=attn_matplotlib,
        title=f"{query_id}: {query_text}",
        cmap_name=cmap_name,
        alpha=alpha,
    )
    render_matplotlib_style(
        thumbnail_np=thumbnail_np,
        heatmap=raw_vis,
        mask=mask,
        save_path=raw_matplotlib,
        title=f"{query_id}: raw score",
        cmap_name=cmap_name,
        alpha=alpha,
    )

    slide.close()
    return attn_overlay


def save_query_defs(query_defs, csv_path):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "query_text"])
        writer.writeheader()
        for query_id, query_text in query_defs:
            writer.writerow({"query_id": query_id, "query_text": query_text})


def save_attention_csv(
    csv_path,
    pair_id,
    label,
    modality,
    coords,
    query_defs,
    attention,
    raw_scores,
):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "pair_id",
            "label",
            "modality",
            "query_id",
            "query_text",
            "patch_name",
            "x",
            "y",
            "col",
            "row",
            "tissue_score",
            "attention_weight",
            "raw_score",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for q_idx, (query_id, query_text) in enumerate(query_defs):
            attn_q = attention[q_idx].numpy()
            raw_q = raw_scores[q_idx].numpy()
            for p_idx, item in enumerate(coords):
                writer.writerow(
                    {
                        "pair_id": pair_id,
                        "label": label,
                        "modality": modality,
                        "query_id": query_id,
                        "query_text": query_text,
                        "patch_name": item["patch_name"],
                        "x": item["x"],
                        "y": item["y"],
                        "col": item["col"],
                        "row": item["row"],
                        "tissue_score": item["tissue_score"],
                        "attention_weight": float(attn_q[p_idx]),
                        "raw_score": float(raw_q[p_idx]),
                    }
                )


def save_modality_summary(csv_path, rows):
    ensure_dir(os.path.dirname(csv_path))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "pair_id",
                "label",
                "query_id",
                "query_text",
                "modality",
                "num_patches",
                "attention_sum",
                "attention_mean",
                "raw_score_mean",
                "raw_score_max",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def render_panel(image_paths, query_defs, save_path, title):
    if not image_paths:
        return

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()
    for ax, path, (query_id, _) in zip(axes, image_paths, query_defs):
        img = Image.open(path).convert("RGB")
        ax.imshow(img)
        ax.set_title(query_id, fontsize=10)
        ax.axis("off")

    for ax in axes[len(image_paths):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def prepare_slide(slide_spec, args):
    slide_name = os.path.splitext(os.path.basename(slide_spec.path))[0]
    slide_out = ensure_dir(
        os.path.join(
            args.out_dir,
            slide_spec.pair_id,
            f"{slide_spec.modality}_{slide_name}",
        )
    )
    patch_dir = ensure_dir(os.path.join(slide_out, "patches"))
    feature_dir = ensure_dir(os.path.join(slide_out, "features"))
    heatmap_dir = ensure_dir(os.path.join(slide_out, "query_heatmaps"))

    coord_csv = os.path.join(slide_out, "patch_coords.csv")
    feature_path = os.path.join(feature_dir, f"{slide_name}_ctranspath_features.pt")

    slide = open_wsi_slide(slide_spec.path)
    w, h = slide.dimensions
    slide.close()

    if os.path.exists(coord_csv) and not args.force_reextract:
        coords = load_patch_coords(coord_csv)
        print(f"[INFO] Reusing patch coords: {coord_csv}")
    else:
        coords, w, h = extract_patches_from_wsi(
            wsi_path=slide_spec.path,
            out_patch_dir=patch_dir,
            tile_size=args.tile_size,
            stride=args.patch_stride or args.tile_size,
            tissue_threshold=args.tissue_threshold,
            stain_type=slide_spec.stain_type,
            ihc_use_dab=args.ihc_use_dab,
            max_patches=args.max_patches_per_slide,
        )
        save_patch_coords(coords, coord_csv)

    return {
        "spec": slide_spec,
        "slide_name": slide_name,
        "slide_out": slide_out,
        "patch_dir": patch_dir,
        "feature_path": feature_path,
        "heatmap_dir": heatmap_dir,
        "coord_csv": coord_csv,
        "coords": coords,
        "w": w,
        "h": h,
    }


def load_or_encode_features(slide_info, encoder, device, args):
    feature_path = slide_info["feature_path"]
    if os.path.exists(feature_path) and not args.force_reencode:
        obj = torch.load(feature_path, map_location="cpu")
        print(f"[INFO] Reusing patch features: {feature_path}")
        return obj["features"].float(), obj["coords"]

    features, valid_coords = encode_patches(
        patch_dir=slide_info["patch_dir"],
        coords=slide_info["coords"],
        encoder=encoder,
        device=device,
        batch_size=args.batch_size,
        use_amp=args.amp,
    )
    torch.save(
        {
            "features": features,
            "coords": valid_coords,
            "feature_dim": int(features.shape[1]),
            "patch_encoder": "ctranspath",
        },
        feature_path,
    )
    print(f"[INFO] Saved patch features: {feature_path}")
    return features, valid_coords


def group_by_pair(slide_infos):
    grouped = {}
    for info in slide_infos:
        pair_id = info["spec"].pair_id
        grouped.setdefault(pair_id, []).append(info)
    return grouped


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Output directory: {args.out_dir}")
    ensure_dir(args.out_dir)

    query_defs = DEFAULT_QUERIES
    save_query_defs(query_defs, os.path.join(args.out_dir, "queries.csv"))

    slides = load_slide_specs(args.slides_csv)
    if args.only_pair:
        slides = [item for item in slides if item.pair_id == args.only_pair]
        if not slides:
            raise ValueError(f"No pair matched --only_pair={args.only_pair}")

    text_tokenizer, text_model, text_max_length = load_text_encoder(
        text_model_name=args.text_model,
        device=device,
        local_files_only=args.local_files_only,
        max_length=args.text_max_length,
    )
    query_embeddings = encode_queries(
        query_defs=query_defs,
        tokenizer=text_tokenizer,
        text_model=text_model,
        max_length=text_max_length,
        device=device,
    )
    print(f"[INFO] Query embeddings: {tuple(query_embeddings.shape)}")

    encoder = load_ctranspath_encoder(args.ctranspath_weight, device)

    slide_infos = []
    for slide_spec in slides:
        print("=" * 100)
        print(f"[INFO] Preparing {slide_spec.pair_id} | {slide_spec.modality}")
        print(f"[INFO] Path: {slide_spec.path}")
        info = prepare_slide(slide_spec, args)
        features, coords = load_or_encode_features(info, encoder, device, args)
        info["features"] = features
        info["coords"] = coords
        slide_infos.append(info)

    image_dim = int(slide_infos[0]["features"].shape[1])
    text_dim = int(query_embeddings.shape[1])
    projection = build_query_projection(
        text_dim=text_dim,
        image_dim=image_dim,
        alignment_ckpt=args.alignment_ckpt,
        device=device,
    )

    summary_rows = []

    if args.attention_scope == "pair":
        grouped = group_by_pair(slide_infos)
        for pair_id, infos in grouped.items():
            print("=" * 100)
            print(f"[INFO] Computing pair-level query attention: {pair_id}")

            features = torch.cat([info["features"] for info in infos], dim=0)
            attention, raw_scores = compute_query_patch_attention(
                patch_features=features,
                query_embeddings=query_embeddings,
                projection=projection,
                device=device,
                temperature=args.temperature,
            )

            pair_out = ensure_dir(os.path.join(args.out_dir, pair_id))
            torch.save(
                {
                    "attention": attention,
                    "raw_scores": raw_scores,
                    "query_defs": query_defs,
                    "attention_scope": "pair",
                },
                os.path.join(pair_out, "pair_query_attention.pt"),
            )

            start = 0
            for info in infos:
                n = len(info["coords"])
                end = start + n
                spec = info["spec"]
                slide_attention = attention[:, start:end]
                slide_raw = raw_scores[:, start:end]

                np.save(os.path.join(info["slide_out"], "query_attention.npy"), slide_attention.numpy())
                np.save(os.path.join(info["slide_out"], "query_raw_scores.npy"), slide_raw.numpy())
                save_attention_csv(
                    csv_path=os.path.join(info["slide_out"], "patch_query_attention.csv"),
                    pair_id=spec.pair_id,
                    label=spec.label,
                    modality=spec.modality,
                    coords=info["coords"],
                    query_defs=query_defs,
                    attention=slide_attention,
                    raw_scores=slide_raw,
                )

                panel_paths = []
                for q_idx, (query_id, query_text) in enumerate(query_defs):
                    overlay_path = draw_query_heatmaps(
                        wsi_path=spec.path,
                        coords=info["coords"],
                        attention_values=slide_attention[q_idx].numpy(),
                        raw_values=slide_raw[q_idx].numpy(),
                        query_id=query_id,
                        query_text=query_text,
                        out_dir=info["heatmap_dir"],
                        tile_size=args.tile_size,
                        downsample=args.heatmap_downsample,
                        cmap_name=args.cmap,
                        alpha=args.alpha,
                        smooth_sigma=args.smooth_sigma,
                    )
                    panel_paths.append(overlay_path)

                    attn_q = slide_attention[q_idx].numpy()
                    raw_q = slide_raw[q_idx].numpy()
                    summary_rows.append(
                        {
                            "pair_id": spec.pair_id,
                            "label": spec.label,
                            "query_id": query_id,
                            "query_text": query_text,
                            "modality": spec.modality,
                            "num_patches": n,
                            "attention_sum": float(attn_q.sum()),
                            "attention_mean": float(attn_q.mean()),
                            "raw_score_mean": float(raw_q.mean()),
                            "raw_score_max": float(raw_q.max()),
                        }
                    )

                if not args.no_panel:
                    render_panel(
                        image_paths=panel_paths,
                        query_defs=query_defs,
                        save_path=os.path.join(info["slide_out"], "all_query_attention_overlay_panel.png"),
                        title=f"{pair_id} | {spec.modality} | label={spec.label}",
                    )

                start = end

    else:
        for info in slide_infos:
            spec = info["spec"]
            print("=" * 100)
            print(f"[INFO] Computing slide-level query attention: {spec.pair_id} | {spec.modality}")
            attention, raw_scores = compute_query_patch_attention(
                patch_features=info["features"],
                query_embeddings=query_embeddings,
                projection=projection,
                device=device,
                temperature=args.temperature,
            )
            np.save(os.path.join(info["slide_out"], "query_attention.npy"), attention.numpy())
            np.save(os.path.join(info["slide_out"], "query_raw_scores.npy"), raw_scores.numpy())
            save_attention_csv(
                csv_path=os.path.join(info["slide_out"], "patch_query_attention.csv"),
                pair_id=spec.pair_id,
                label=spec.label,
                modality=spec.modality,
                coords=info["coords"],
                query_defs=query_defs,
                attention=attention,
                raw_scores=raw_scores,
            )

            panel_paths = []
            for q_idx, (query_id, query_text) in enumerate(query_defs):
                overlay_path = draw_query_heatmaps(
                    wsi_path=spec.path,
                    coords=info["coords"],
                    attention_values=attention[q_idx].numpy(),
                    raw_values=raw_scores[q_idx].numpy(),
                    query_id=query_id,
                    query_text=query_text,
                    out_dir=info["heatmap_dir"],
                    tile_size=args.tile_size,
                    downsample=args.heatmap_downsample,
                    cmap_name=args.cmap,
                    alpha=args.alpha,
                    smooth_sigma=args.smooth_sigma,
                )
                panel_paths.append(overlay_path)

                attn_q = attention[q_idx].numpy()
                raw_q = raw_scores[q_idx].numpy()
                summary_rows.append(
                    {
                        "pair_id": spec.pair_id,
                        "label": spec.label,
                        "query_id": query_id,
                        "query_text": query_text,
                        "modality": spec.modality,
                        "num_patches": len(info["coords"]),
                        "attention_sum": float(attn_q.sum()),
                        "attention_mean": float(attn_q.mean()),
                        "raw_score_mean": float(raw_q.mean()),
                        "raw_score_max": float(raw_q.max()),
                    }
                )

            if not args.no_panel:
                render_panel(
                    image_paths=panel_paths,
                    query_defs=query_defs,
                    save_path=os.path.join(info["slide_out"], "all_query_attention_overlay_panel.png"),
                    title=f"{spec.pair_id} | {spec.modality} | label={spec.label}",
                )

    save_modality_summary(
        os.path.join(args.out_dir, "query_attention_modality_summary.csv"),
        summary_rows,
    )
    print("=" * 100)
    print("[DONE] Query attention heatmaps finished.")
    print(f"[DONE] Results saved to: {args.out_dir}")


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Extract small WSI patches, encode them with CHIEF CTransPath, "
            "match six ClinicalBERT query embeddings to patches, and save query heatmaps."
        )
    )
    parser.add_argument(
        "--slides_csv",
        type=str,
        required=True,
        help="CSV manifest with pair_id,label,modality,path,stain_type columns.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="./query_attention_heatmap_outputs",
        help="Output root directory.",
    )
    parser.add_argument(
        "--ctranspath_weight",
        type=str,
        default="./model_weight/CHIEF_CTransPath.pth",
        help="Path to CHIEF_CTransPath.pth.",
    )
    parser.add_argument(
        "--text_model",
        type=str,
        default="emilyalsentzer/Bio_ClinicalBERT",
        help="ClinicalBERT checkpoint name or local directory.",
    )
    parser.add_argument(
        "--alignment_ckpt",
        type=str,
        default="",
        help=(
            "Optional trained text-to-image projection checkpoint. Without it, "
            "the script uses identity mapping when text_dim == image_dim."
        ),
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Use only locally cached Hugging Face files.",
    )
    parser.add_argument(
        "--only_pair",
        type=str,
        default="",
        help="Optional pair_id to run only one default HE-IHC pair.",
    )
    parser.add_argument(
        "--attention_scope",
        type=str,
        default="pair",
        choices=["pair", "slide"],
        help="pair: softmax over HE+IHC patches in the same pair. slide: softmax per slide.",
    )
    parser.add_argument(
        "--tile_size",
        type=int,
        default=80,
        help="Patch size at level-0. Smaller gives smoother visual heatmaps but more patches.",
    )
    parser.add_argument(
        "--patch_stride",
        type=int,
        default=0,
        help="Patch stride at level-0. Default 0 means stride = tile_size.",
    )
    parser.add_argument(
        "--tissue_threshold",
        type=float,
        default=0.05,
        help="Background filtering threshold. Larger means stricter filtering.",
    )
    parser.add_argument(
        "--max_patches_per_slide",
        type=int,
        default=0,
        help="Debug limit. 0 means use all tissue patches.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Patch encoding batch size.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Softmax temperature for query-patch similarity.",
    )
    parser.add_argument(
        "--text_max_length",
        type=int,
        default=64,
        help="Max token length for ClinicalBERT queries.",
    )
    parser.add_argument(
        "--heatmap_downsample",
        type=int,
        default=32,
        help="Downsample factor for WSI heatmap thumbnail.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="rainbow",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.55,
        help="Overlay alpha for heatmap on WSI thumbnail.",
    )
    parser.add_argument(
        "--smooth_sigma",
        type=float,
        default=1.5,
        help="Gaussian smoothing sigma on thumbnail-scale heatmap. 0 disables smoothing.",
    )
    parser.add_argument(
        "--ihc_use_dab",
        action="store_true",
        help="Use DAB-like grayscale RGB patches for IHC encoder input.",
    )
    parser.add_argument(
        "--force_reextract",
        action="store_true",
        help="Re-extract patches even if patch_coords.csv exists.",
    )
    parser.add_argument(
        "--force_reencode",
        action="store_true",
        help="Recompute CTransPath features even if cached features exist.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use CUDA autocast for faster patch encoding.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU.",
    )
    parser.add_argument(
        "--no_panel",
        action="store_true",
        help="Do not save the 2x3 all-query overlay panel.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
