"""Training-only, channel-specific scaling of raw stain concentration means."""

import hashlib
import json
import math
import os

import numpy as np

SCALAR_SCHEMA = "raw_concentration_mean_v2"
CHANNELS = ("he_h", "ihc_h", "ihc_dab")
MODE = "train_standardized"


def validate_scaler(scaler):
    if not isinstance(scaler, dict) or scaler.get("schema") != SCALAR_SCHEMA:
        raise ValueError("A fitted raw-concentration stain scaler is required")
    if scaler.get("method") != "zscore_population_no_clip":
        raise ValueError("Unsupported stain scaling method")
    for channel in CHANNELS:
        s = scaler.get("channels", {}).get(channel, {})
        if (int(s.get("count", 0)) <= 0 or not math.isfinite(s.get("mean", float("nan")))
                or not math.isfinite(s.get("scale", float("nan"))) or s["scale"] <= 0):
            raise ValueError("Invalid stain scaler channel: " + channel)
    return scaler


def fit_scaler(records, training_fingerprint):
    """Records are (channel, stable unique tile key, raw concentration mean).

    Each distinct training-slide tile contributes once per channel, even if a
    slide occurs in several pairs. Population moments use float64 Welford sums.
    No labels, validation values, test values, or clipping are used.
    """
    state = {c: [0, 0.0, 0.0] for c in CHANNELS}
    seen = {c: set() for c in CHANNELS}
    for channel, tile_key, value in records:
        if channel not in state:
            raise ValueError("Unknown stain channel: " + str(channel))
        value = float(value)
        if not math.isfinite(value) or value < 0:
            raise ValueError("Raw concentration means must be finite and nonnegative")
        if tile_key in seen[channel]:
            continue
        seen[channel].add(tile_key)
        n, mean, m2 = state[channel]
        n += 1
        delta = value - mean
        mean += delta / n
        m2 += delta * (value - mean)
        state[channel] = [n, mean, m2]
    channels = {}
    for channel, (n, mean, m2) in state.items():
        if n == 0:
            raise ValueError("No training concentrations for " + channel)
        std = math.sqrt(max(m2 / n, 0.0))
        channels[channel] = dict(count=n, mean=mean, std=std,
                                 scale=std if std > 1e-8 else 1.0)
    return validate_scaler(dict(schema=SCALAR_SCHEMA,
                                method="zscore_population_no_clip",
                                training_fingerprint=training_fingerprint,
                                channels=channels))


def scale_values(values, channel, scaler):
    """Works on NumPy arrays or torch tensors, preserving backend/device."""
    validate_scaler(scaler)
    if channel not in CHANNELS:
        raise ValueError("Unknown stain channel: " + str(channel))
    stats = scaler["channels"][channel]
    return (values - stats["mean"]) / stats["scale"]


def scale_case(case, scaler):
    if case.get("stain_scalar_schema") != SCALAR_SCHEMA:
        raise ValueError("Case cache does not contain raw concentration means")
    result = dict(case)
    for channel in CHANNELS:
        result[channel] = scale_values(case[channel], channel, scaler)
    return result


def prepare_stain_scaler(args, train_df, build_cache, cache_dir):
    """Fit strictly after the actual train/validation/test split is resolved."""
    if getattr(args, "stain_scalar_mode", MODE) != MODE:
        args.stain_scaler = None
        return
    if args.morphology_mode != "macenko":
        raise ValueError("train_standardized stain scalars require morphology_mode=macenko")
    if getattr(args, "allow_missing_morphology", False):
        raise ValueError("Raw concentration scaling requires all source RGB patches")
    if train_df.empty:
        raise ValueError("Cannot fit a stain scaler without training data")
    identity = sorted((str(r["project_name"]), str(r["case_id"]),
                       str(r["pair_case_name"])) for _, r in train_df.iterrows())
    fingerprint = hashlib.sha256(json.dumps(identity).encode()).hexdigest()

    def records():
        for _, row in train_df.sort_values(["project_name", "pair_case_name"]).iterrows():
            case = build_cache(row, args, cache_dir)
            if case.get("stain_scalar_schema") != SCALAR_SCHEMA:
                raise ValueError("Unexpected cache schema during training-scaler fit")
            for channel in CHANNELS:
                paths = case["he_patch_paths" if channel == "he_h" else "ihc_patch_paths"]
                values = case[channel]
                if len(paths) != len(values):
                    raise ValueError("Concentration/path count mismatch")
                for path, value in zip(paths, values):
                    if not path:
                        raise ValueError("Raw concentration means need original RGB paths")
                    yield channel, os.path.realpath(path), float(value)

    args.stain_scaler = fit_scaler(records(), fingerprint)
    os.makedirs(args.results_dir, exist_ok=True)
    path = os.path.join(args.results_dir, "stain_scaler.json")
    with open(path, "w") as handle:
        json.dump(args.stain_scaler, handle, indent=2, allow_nan=False)
    print("[INFO] Fitted stain scaler on unique training tiles only: " + path)
