"""Exact bounded spiral search. Copy next to train_vgms.py.

The exhaustive control uses the SAME bounded FP32 similarity and FP64 logits.
Unvisited diagnostic entries are NaN, not measured similarity values.
"""
import math
import os
import torch


def bounded_spiral_align(model, he_features, ihc_features, he_h_map, ihc_h_map,
                         he_yx, ihc_yx, order_fn):
    n, m = len(ihc_yx), len(he_yx)
    if n == 0 or m == 0:
        raise ValueError("Alignment requires nonempty anchors and candidates")
    gamma, sigma, stride = model.spatial_gamma, model.spatial_sigma, model.patch_stride
    if not all(math.isfinite(x) for x in (gamma, sigma, stride, model.temperature)):
        raise ValueError("Non-finite alignment configuration")
    if gamma < 0 or sigma <= 0 or stride <= 0 or model.temperature <= 0:
        raise ValueError("Invalid alignment configuration")
    k = min(model.align_topk, m)
    if k < 1:
        raise ValueError("align_topk must be positive")
    enabled = os.environ.get("VGMS_SAFE_STOP", "1") != "0"
    batch = int(os.environ.get("VGMS_SEARCH_BATCH", "128"))
    if batch < 1:
        raise ValueError("VGMS_SEARCH_BATCH must be positive")
    device = he_features.device
    he_coords = torch.tensor(he_yx, dtype=torch.float64, device=device) / stride
    ihc_coords = torch.tensor(ihc_yx, dtype=torch.float64, device=device) / stride
    # Explicit differences avoid cdist's matrix-multiplication cancellation.
    distance2 = (ihc_coords[:, None] - he_coords[None]).square().sum(-1)
    penalty = distance2 / (2 * sigma ** 2)
    deduction = gamma * penalty
    if not torch.isfinite(deduction).all():
        raise ValueError("Non-finite coordinates or spatial penalty")
    sf = torch.full((n, m), float("nan"), dtype=torch.float32, device=device)
    sm, sb = sf.clone(), sf.clone()
    logits = torch.full((n, m), -float("inf"), dtype=torch.float64, device=device)
    evaluated = torch.zeros((n, m), dtype=torch.bool, device=device)
    counts = []
    for i, origin in enumerate(ihc_yx):
        order = torch.tensor(order_fn(he_yx, origin), dtype=torch.long, device=device)
        # Exact suffix maximum over ALL unvisited coordinate-based upper bounds.
        upper = 1.0 - deduction[i, order]
        suffix = torch.flip(torch.cummax(torch.flip(upper, [0]), 0).values, [0])
        count = 0
        for start in range(0, m, batch):
            ids = order[start:start + batch]
            # Disable outer AMP for stable bounded scoring in both modes.
            with torch.autocast(device_type=device.type, enabled=False):
                f, h, b, mix = model._similarity(
                    he_features[ids].float(), ihc_features[i:i+1].float(),
                    he_h_map[ids].float(), ihc_h_map[i:i+1].float())
            if not all(torch.isfinite(x).all() for x in (f, h, b)):
                raise ValueError("Non-finite similarity")
            # Mathematical range enforced against numerical overshoot.
            bounded = b[0].float().clamp(-1.0, 1.0)
            sf[i, ids], sm[i, ids], sb[i, ids] = f[0], h[0], bounded
            logits[i, ids] = bounded.double() - deduction[i, ids]
            evaluated[i, ids] = True
            count += len(ids)
            if enabled and count >= k and count < m:
                threshold = torch.topk(logits[i], k).values[-1]
                # Strict inequality preserves original-index tie breaking.
                margin = 1e-12 * (1 + threshold.abs() + suffix[count].abs())
                if bool(suffix[count] + margin < threshold):
                    break
        counts.append(count)
    indices = torch.argsort(logits, dim=1, descending=True, stable=True)[:, :k]
    top = torch.gather(logits, 1, indices)
    weights = torch.softmax(top / model.temperature, dim=1).to(he_features.dtype)
    selected = he_features[indices]
    aligned = torch.einsum("nk,nkd->nd", weights, selected)
    model.last_search_stats = {
        "safe_stop": enabled, "evaluated_per_anchor": counts,
        "candidates_per_anchor": m, "evaluated_pairs": sum(counts),
        "total_pairs": n * m, "scored_fraction": sum(counts) / (n * m),
    }
    model.last_alignment_evaluated_mask = evaluated
    return (aligned, indices, weights, sf, sm, sb,
            logits.masked_fill(~evaluated, float("nan")),
            distance2.sqrt(), penalty, mix)
