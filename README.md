# VGMS

Reference implementation for:

**VISION-LANGUAGE WEIGHT-GUIDED IMMUNOTHERAPY RESPONSE PREDICTION FOR NSCLC USING MULTI-STAIN WHOLE-SLIDE IMAGES**

VGMS integrates paired hematoxylin and eosin (H&E) and PD-L1 immunohistochemistry (IHC) whole-slide image representations for pathological complete response prediction in non-small cell lung cancer. The repository contains the main training pipeline, single-stain baselines, module and encoder ablations, cohort visualization, result summarization, and query-attention heatmap generation.

The default configuration follows the reported protocol. For each sampled IHC anchor, all retained H&E candidates from the same case are eligible for matching using hematoxylin-channel SSIM, CHIEF feature cosine similarity, and a Gaussian spatial-distance penalty. The highest-scoring 64 candidates are softmax weighted to aggregate both the H&E embedding and hematoxylin intensity. The aligned H&E representation is concatenated with the IHC CHIEF embedding and IHC DAB intensity, then projected. Six frozen ClinicalBERT query embeddings attend to the aligned visual tokens, and the six query-specific evidence vectors are concatenated for a three-layer gated MLP classifier.

## Repository contents

| File | Purpose |
|---|---|
| `train_vgms.py` | Main multi-stain query-guided MIL training and evaluation pipeline |
| `train_single_stain_baselines.py` | H&E-only and IHC-only CHIEF baselines |
| `run_module_ablation.py` | Input, alignment, and vision-language weighting ablations |
| `run_text_encoder_ablation.py` | ClinicalBERT, TF-IDF, CLIP, and Qwen2.5-0.5B comparisons |
| `run_image_encoder_ablation.py` | CHIEF, Virchow2, UNI-2h, and TITAN feature comparisons |
| `visualize_query_attention.py` | Query-specific WSI attention heatmaps |
| `plot_cohort_statistics.py` | De-identified cohort summary visualization |
| `summary_utils.py` | Shared metric and summary-table utilities |
| `summarize_results.py` | Aggregate completed experiment outputs |
| `verify_paper_protocol.py` | Static check for paper-aligned implementation defaults |
| `smoke_test.py` | CPU forward/backward, alignment-shape, and patient-split smoke test |
| `select_alignment_lambda.py` | Validation-only grid search for the fixed alignment mixture |

## Study cohorts

The study uses retrospectively collected data from two hospitals. All splits are patient-level. 

### Full study population

| Cohort | Role | Patients | Paired H&E-IHC images |
|---|---:|---:|---:|
| Cohort 1 | Development and internal evaluation | 263 | 337 |
| Cohort 2 | Independent external evaluation | 100 | 101 |
| Total | - | 363 | 438 |

### Evaluation subsets

| Evaluation subset | Patients | H&E slides | PD-L1 IHC slides | H&E-IHC image pairs |
|---|---:|---:|---:|---:|
| Cohort 1 internal test | 80 | 96 | 80 | 96 |
| Cohort 2 external test | 100 | 101 | 101 | 101 |


## Data availability

The source WSIs and associated clinical annotations cannot be made publicly available because they contain protected patient information and are governed by institutional ethics approvals and data-use restrictions. 

## Environment

The experiments were developed with Python 3.11, PyTorch 2.1.0, CUDA 12.1, and NVIDIA RTX 4090 GPUs.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Pretrained model weights are not redistributed here. Hugging Face checkpoints are downloaded automatically on first use unless `--local_files_only` is supplied. Access-controlled pathology encoders must be obtained from their original providers.

## Expected metadata

The main pipeline expects de-identified CSV files containing the following fields:

```text
case_id,he_file,ihc_file,pair_case_name,label
```

`case_id` is used to keep all image pairs from one patient in the same split. `label` is binary, with `1` indicating pCR and `0` indicating non-pCR. The `*.example.csv` manifests use synthetic identifiers only.

A representative feature layout is:

```text
data/patches/
  cohort1/
    pair_labels_with_case_id.csv
    train_split.csv
    test_split.csv
    FusionFeatures_patch_level_tiles/
  cohort2/
    pair_labels_with_case_id.csv
    FusionFeatures_patch_level_tiles/
```

## Main experiment

```bash
python train_vgms.py \
  --project_name cohort1 \
  --external_project_name cohort2 \
  --base_path ./data/patches \
  --train_split_csv ./data/patches/cohort1/train_split.csv \
  --test_split_csv ./data/patches/cohort1/test_split.csv \
  --results_dir ./results/vgms_main \
  --val_size 0.2857142857 \
  --epochs 80 \
  --patience 10 \
  --lr 5e-5 \
  --loss focal \
  --morphology_mode macenko \
  --alignment_lambda 0.5 \
  --align_topk 64 \
  --align_temperature 1.0 \
  --spatial_gamma 0.25 \
  --spatial_sigma 8.0 \
  --max_patches_per_case 1024 \
  --n_bootstraps 2000 \
  --prob_postprocess temperature \
  --prob_temperature 4 \
  --device cuda:0 \
  --amp --tf32
```

Checkpoint selection and early stopping use only the validation subset derived from the training split.

### Protocol defaults

| Setting | Value |
|---|---:|
| Patch size and magnification | 250 x 250 pixels at 40x |
| Split | Patient-disjoint 5:2:3 train/validation/internal test |
| IHC anchors sampled per image pair | At most 1,024 |
| H&E matching candidates | All retained H&E patches from the same case |
| Image features | Frozen CHIEF patch embeddings |
| Stain-channel inputs | Scalar mean H&E hematoxylin and IHC DAB intensities |
| SSIM input | Macenko hematoxylin channel resized to `32 x 32` |
| Text encoder | Frozen Bio_ClinicalBERT checkpoint |
| Alignment | Distance-aware Top-64 soft aggregation; temperature 1; `gamma=0.25`; `sigma=8` patch units; fixed `alignment_lambda` selected on validation AUC |
| Optimizer | AdamW, learning rate `5e-5`, weight decay `1e-4` |
| Schedule | Cosine decay to `1e-6`, at most 80 epochs |
| Loss | Binary focal loss, gamma 2, training-label-derived class weights |
| Early stopping | Validation AUC, patience 10 |
| Saved prediction probabilities | Temperature scaling with `T=4`; applied only when writing final prediction files |
| Decision threshold | 0.5 |
| AUC confidence interval | 2,000 patient-cluster bootstrap resamples retaining pair predictions, percentile limits, seed 42 |

The default `alignment_lambda=0.5` is a reproducible placeholder. Run the validation-only selection workflow below to implement the paper's grid search. Candidate runs skip both test cohorts; after selection, the script trains and evaluates one final model with the selected value.

```bash
python select_alignment_lambda.py \
  --project_name cohort1 \
  --external_project_name cohort2 \
  --base_path ./data/patches \
  --train_split_csv ./data/patches/cohort1/train_split.csv \
  --test_split_csv ./data/patches/cohort1/test_split.csv \
  --lambdas 0 0.25 0.5 0.75 1 \
  --device cuda:0
```

The selected value, all validation AUCs, and the final test results are saved under `results/alignment_lambda_selection/`. The selected numerical value should also be stated in the manuscript for exact reproducibility.

For IHC anchor `i` and H&E candidate `j`, the distance-aware alignment logit is

```math
\ell_{ij}=\lambda s^{\mathrm{morph}}_{ij}+(1-\lambda)s^{\mathrm{feat}}_{ij}-\gamma\frac{\lVert \mathbf{p}^{I}_{i}-\mathbf{p}^{H}_{j}\rVert_2^2}{2\sigma^2}
```

where coordinates are converted from common-frame pixels to patch-grid units. The default values are `gamma=0.25` and `sigma=8`; `lambda` is selected using validation AUC only. Let `N_i=TopK_j(ell_ij, K=64)`. The aggregation weights and aligned H&E representation are

```math
w_{ij}=\frac{\exp(\ell_{ij}/\tau)}{\sum_{k\in\mathcal{N}_i}\exp(\ell_{ik}/\tau)},\qquad
\tilde{\mathbf{v}}^H_i=\sum_{j\in\mathcal{N}_i}w_{ij}\mathbf{v}^H_j
```

with `tau=1`. The same weights aggregate H&E hematoxylin intensity.

Matching starts at each IHC anchor's corresponding location in the shared, already scale-mapped coordinate frame. Tile names are `y_x.png` (vertical then horizontal coordinate), with matching feature stems `y_x.pt`. Candidates are visited in clockwise expanding square rings in batches of at most 128. Safe stopping uses the maximum spatial score upper bound over all unvisited candidates. Search stops only when this bound plus a numerical margin is strictly below the current Kth score. Otherwise it continues exhaustively. Spatial locality is imposed by the distance term, not by traversal order. Top-64 selection and softmax weighting both use `ell_ij`, with ties resolved by the lowest original candidate index. Existing caches remain usable because coordinates are parsed from retained tile names. Invalid coordinate names raise an error.

## Safe stopping and exhaustive control

Safe stopping is enabled by default. Prefix the normal command with `VGMS_SAFE_STOP=0` to score every candidate, or `VGMS_SAFE_STOP=1` for bounded search. `VGMS_SEARCH_BATCH` defaults to 128. Run `python test_safe_spiral.py` to compare indices, weights, outputs and gradients.

The similarity mixture is bounded to [-1,1] and evaluated in FP32 outside AMP. Coordinates, penalties and logits use FP64. Both modes use identical new arithmetic. Bitwise equivalence to the previous AMP/FP32-cdist implementation is not guaranteed. CPU tests passed, while CUDA/AMP and real-checkpoint validation remain required. Compare the same checkpoint and inputs before training new experiments. No wall-clock speedup is claimed.

Unscored diagnostic matrix entries are NaN and are identified by `alignment_evaluated_mask` in intermediate NPZ files. `model.last_search_stats` reports the last forward pass only. Dense diagnostics remain, so memory reduction is not claimed.

## Reproducibility and privacy

- Random seed defaults to 42.
- Test cohorts are evaluation-only.
- Generated predictions retain de-identified case and slide keys for paired statistical analysis.
- `data/`, model weights, checkpoints, caches, predictions, and WSI formats are excluded by `.gitignore`.
- Run `python privacy_scan.py .` before every release commit.
- Run `python verify_paper_protocol.py` to check paper-protocol defaults and core implementation markers.
- Run `python smoke_test.py` after installing dependencies to validate model shapes, gradients, and patient-disjoint splitting.

