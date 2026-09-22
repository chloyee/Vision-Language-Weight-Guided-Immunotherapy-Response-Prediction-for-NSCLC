# Stain maps and fusion scalars

The default `--stain_scalar_mode train_standardized` separates two inputs:

- **Structural matching:** patchwise min-max normalized hematoxylin maps, resized to 32 x 32, are used for SSIM.
- **Fusion scalars:** spatial means of the nonnegative concentration maps are computed before patchwise normalization. Each scalar channel is standardized using only the final training subset.

For channel c, the transform is `(raw_mean - training_mean[c]) / training_std[c]`. Channels are H&E hematoxylin (`he_h`), IHC hematoxylin (`ihc_h`, used for IHC-only inputs), and IHC DAB (`ihc_dab`). Statistics use population standard deviations and every unique training RGB tile once per channel. For a constant channel (standard deviation <= 1e-8), the scale is one. Values are not clipped. Validation and test values never contribute to the fitted statistics.

The SSIM maps and their descriptors retain their patch-relative representation. Raw concentration means are not replaced by precomputed DAB PNG intensities. Source RGB patches are required in the default mode. These numerical concentrations are not calibrated protein-expression measurements.

## Training and inference

Normal `train_vgms.py` commands automatically fit the scaler after patient-disjoint splitting, before constructing the data loaders. Module and encoder ablation runners call the same pipeline. `--cache_only` also fits the training scaler.

`stain_scaler.json` stores the fitted values, channel counts, method, and a training-membership fingerprint. The same object is stored in `config.json` and in the checkpoint's `args.stain_scaler`. For inference on a saved checkpoint, restore its arguments, including `stain_scaler`, and use `CaseMILDataset` or apply `stain_scaling.scale_case` to raw caches. Do not call the fitting function on inference data.

Caches are named by scalar mode and schema. They contain raw concentration means, not standardized values, so the loader applies the fitted transform exactly once. The default loader rejects caches without the expected schema and rejects a missing scaler.

`--stain_scalar_mode legacy_patch_mean` is an explicit alternative that uses patch-relative scalar means. `fast_gray` and `none` morphology modes are supported only with that scalar mode. The default concentration mode uses `--morphology_mode macenko` and requires all RGB patches.

Run `python test_stain_scaling.py` for the synthetic scaling checks. Pretrained image features, text embeddings, alignment scoring, and spatial safe stopping are independent of scalar standardization.
