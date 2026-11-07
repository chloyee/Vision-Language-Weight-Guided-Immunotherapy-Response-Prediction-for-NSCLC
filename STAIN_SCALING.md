# Stain maps and fusion scalars

VGMS uses complementary representations of stain concentration for structural correspondence and visual fusion.

## Structural maps

Macenko stain vectors define a two-channel optical-density basis. Nonnegative least squares estimates the concentration at each pixel. Patchwise min-max normalization of hematoxylin concentrations produces maps in [0, 1], resized to 32 x 32 for SSIM.

## Fusion scalars

Spatial means are computed from concentration maps before patchwise normalization. With the default train_standardized mode, each channel is transformed as:

    standardized_mean = (raw_mean - training_mean) / training_std

The channels are H&E hematoxylin (he_h), IHC hematoxylin (ihc_h for IHC-only inputs), and IHC DAB (ihc_dab). Each unique training RGB tile contributes once per channel. Population moments determine the mean and standard deviation. A constant channel uses scale one. The transform retains values outside [0, 1].

The full model aggregates standardized H&E hematoxylin means with the same softmax weights as H&E features, then concatenates them with IHC features and standardized DAB means. Source RGB patches supply the concentration measurements.

## Training and checkpoint reuse

The training pipeline fits channel statistics after resolving patient-disjoint partitions. Validation and evaluation use the fixed training transform. The cache-only command also computes these statistics.

The fitted channel counts, means, standard deviations, and scales are saved in stain_scaler.json, config.json, and the checkpoint configuration. Case caches store raw means. CaseMILDataset applies the fitted transform when reading a sample.

For checkpoint reuse, restore args.stain_scaler from the checkpoint and construct CaseMILDataset with those arguments. The scale_case function also applies the transform directly to a raw case cache. Retain the checkpoint's statistics for inference.

## Options

- --stain_scalar_mode train_standardized and --morphology_mode macenko select concentration-based fusion.
- --stain_scalar_mode legacy_patch_mean selects patch-relative scalar means. This mode also supports fast_gray and none morphology settings.
- --cache_only prepares caches and training statistics.

Run python test_stain_scaling.py to check scalar fitting and application.
