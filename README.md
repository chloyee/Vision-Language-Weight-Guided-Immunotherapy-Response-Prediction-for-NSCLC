# VGMS

**Vision-Language Weight-Guided Immunotherapy Response Prediction for NSCLC Using Multi-Stain Whole-Slide Images**

VGMS combines cross-stain correspondence with pathology-guided aggregation for pathological complete response prediction. Spatially regularized structural and feature similarity assembles H&E evidence around IHC anchors. Six pathology queries then pool the aligned representations into immune, stromal, and PD-L1 evidence for a gated classifier.

## Installation

Use Python 3.11.4 with PyTorch 2.1.0 and CUDA 12.1. The experimental hardware consists of two NVIDIA RTX 4090 GPUs. Each training process selects one device through `--device`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

## Data preparation

Training starts from color-normalized RGB tissue patches and frozen CHIEF features. Prepare non-overlapping 250 x 250 patches at 40x. Map slide coordinates into a shared frame using the H&E and IHC slide dimensions. Name patches `y_x.png` and feature files `y_x.pt`, where y and x are vertical and horizontal **pixel coordinates in that shared frame**. The model divides these coordinates by 250 when computing distances.

For the `fusion_tiles` input format, each feature file contains a float tensor of shape `[2, 768]`, with H&E in row 0 and IHC in row 1. A flattened `[1536]` tensor is also supported. Each retained feature stem has a corresponding RGB patch in both patch directories. All retained H&E rows form the candidate bank. The loader samples at most 1,024 IHC rows for each image pair.

Organize prepared inputs as follows:

```text
data/patches/
  cohort1/
    pair_labels_with_case_id.csv
    train_split.csv
    test_split.csv
    FusionFeatures_patch_level_tiles/
      example_he_001__example_ihc_001/
        0_0.pt
        0_250.pt
    patches_lv0_ps250/
      cohort1_HE_slides_data/example_he_001/
        0_0.png
        0_250.png
      cohort1_IHC_slides_data/example_ihc_001/
        0_0.png
        0_250.png
  cohort2/
    pair_labels_with_case_id.csv
    FusionFeatures_patch_level_tiles/
    patches_lv0_ps250/
```

The cohort metadata has one row per image pair:

```csv
case_id,he_file,ihc_file,pair_case_name,label
example_patient_001,example_he_001.svs,example_ihc_001.svs,example_he_001__example_ihc_001,1
```

- `case_id` identifies a patient. All pairs belonging to a patient share this identifier.
- `label` is 1 for pCR and 0 for non-pCR.
- `he_file` and `ihc_file` identify the source slides.
- `pair_case_name` identifies the feature directory.
- Optional `he_patches_path` and `ihc_patches_path` fields specify alternative RGB patch directories.

Fixed split CSVs contain `case_id,he_file,ihc_file,label`. `train_split.csv` is the **70% development pool**, from which 2/7 of patients form validation and 5/7 form training. `test_split.csv` contains the remaining 30% of patients. With no fixed split files, the loader creates the same 5:2:3 response-stratified patient partition. All image pairs from a patient remain together. The files `train_split.example.csv` and `test_split.example.csv` illustrate the schema with synthetic identifiers.

The training entry point consumes these prepared patches and embeddings. Obtain the CHIEF encoder from its original distribution and preserve the feature-to-patch correspondence during extraction. Bio_ClinicalBERT is loaded from `emilyalsentzer/Bio_ClinicalBERT`, or supply its six query embeddings as a `[6, d]` NumPy array through `--query_embedding_path`. Model access is managed through the model providers.

### Cohorts and evaluation units

| Cohort | Role | Patients | H&E-IHC pairs |
|---|---|---:|---:|
| GDPH | Development and internal evaluation | 263 | 337 |
| FAHZU | External evaluation | 100 | 101 |
| Total | | 363 | 438 |

| Evaluation subset | Patients | H&E slides | IHC slides | Pairs |
|---|---:|---:|---:|---:|
| GDPH internal test | 80 | 96 | 80 | 96 |
| FAHZU external test | 100 | 101 | 101 | 101 |

Multi-stain prediction uses image pairs. Single-stain evaluation uses distinct WSIs. Source WSIs and clinical annotations are governed by institutional data-access and ethics requirements. Training accepts locally prepared, de-identified data in the format above.

## Training

The alignment mixture is selected by GDPH validation AUC. The following command runs a configurable candidate grid, selects its best validation result, and trains the final model with the selected mixture:

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

The grid is configurable with `--lambdas`. Candidate runs train and select checkpoints on the development subsets. The final run evaluates the internal and external cohorts. Outputs include `alignment_lambda_grid.csv`, `selected_alignment_lambda.json`, and `final_selected_model/`.

To train directly with a selected mixture, assign that value to `ALIGNMENT_LAMBDA` and run:

```bash
python train_vgms.py \
  --project_name cohort1 \
  --external_project_name cohort2 \
  --base_path ./data/patches \
  --train_split_csv ./data/patches/cohort1/train_split.csv \
  --test_split_csv ./data/patches/cohort1/test_split.csv \
  --results_dir ./results/vgms_main \
  --alignment_lambda "${ALIGNMENT_LAMBDA:?Set the validation-selected mixture}" \
  --stain_scalar_mode train_standardized \
  --morphology_mode macenko \
  --epochs 80 --patience 10 --lr 5e-5 \
  --loss focal --focal_gamma 2 \
  --align_topk 64 --align_temperature 1 \
  --spatial_gamma 0.25 --spatial_sigma 8 \
  --patch_size 250 --ssim_map_size 32 \
  --max_patches_per_case 1024 \
  --n_bootstraps 2000 \
  --prob_postprocess temperature --prob_temperature 4 \
  --device cuda:0
```

The default seed is 42. The image encoder supplies frozen features and the text encoder supplies frozen embeddings. Visual/query projections, cross-attention parameters, and the classifier are optimized. Checkpoint selection maximizes validation AUC with patience 10.

### Configuration

| Setting | Value |
|---|---|
| Image and text encoders | CHIEF and Bio_ClinicalBERT |
| Anchors | At most 1,024 IHC patches per pair |
| Anchor sampling | Random during training, deterministic during evaluation |
| Similarities | Whole-map hematoxylin SSIM and feature cosine similarity |
| SSIM | 32 x 32 maps in [0, 1], sample statistics, constants 0.01 squared and 0.03 squared |
| Spatial penalty | Weight 0.25, scale 8 patch units |
| Soft aggregation | Top min(64, candidate count), temperature 1 |
| Stain scalars | Raw concentration means, channel-specific training standardization |
| Queries | Six pathology descriptions in `DEFAULT_QUERIES` |
| Classifier | Three-layer gated MLP |
| Optimization | AdamW, initial learning rate 5e-5, at most 80 epochs |
| Loss | Binary focal loss, gamma 2, class weights from training labels |
| Probability | sigmoid(logit / 4), threshold 0.5 |
| Evaluation | Accuracy, positive-class binary F1, AUC |
| AUC intervals | 2,000 patient-cluster resamples, percentile 95% limits, retaining pair predictions |

Additional implementation defaults are hidden dimension 256, dropout 0.35, weight decay 1e-4, cosine learning-rate decay to 1e-6, gradient clipping at 5, and one variable-length image pair per optimizer step. `python train_vgms.py --help` lists all options.

## Alignment and stain features

The alignment score combines morphology and feature similarity with a spatial-distance penalty. Softmax over the selected candidates weights both H&E embeddings and H&E hematoxylin scalars. These values are concatenated with the IHC embedding and DAB scalar before visual projection.

Clockwise expanding square rings order the candidate search. For each unvisited candidate, one minus its weighted spatial penalty bounds its score. Search stops when the largest remaining bound, including a numerical margin, falls below the current Kth score. Ties are resolved by original candidate index. Multiple IHC anchors can share H&E candidates.

`VGMS_SAFE_STOP=1` enables bounded search. `VGMS_SAFE_STOP=0` evaluates all candidates with the same scoring arithmetic. `VGMS_SEARCH_BATCH` sets the search batch size, which defaults to 128. Similarity is evaluated in FP32 and spatial scores in FP64. Intermediate arrays include `alignment_evaluated_mask` to identify evaluated entries.

[Stain maps and fusion scalars](STAIN_SCALING.md) describes concentration decomposition, training statistics, and checkpoint reuse.

## Outputs

Each result directory contains:

- `best_model.pth`: selected model, configuration, query definitions and embeddings.
- `config.json` and `stain_scaler.json`: configuration and fitted channel statistics.
- `splits/`: resolved image-pair partitions.
- `queries.csv` and `query_embeddings.npy`: query inputs.
- `*_wsi_predictions.csv`: image-pair probabilities and labels.
- `*_wsi_metrics.csv`: classification metrics and AUC intervals.
- `intermediates/`: alignment indices, aggregation weights, query attention, and feature representations.
- `summary_2dp.csv` and `summary_4dp.csv`: formatted result summaries.

## Checks and utilities

```bash
python verify_paper_protocol.py
python smoke_test.py
python -m unittest -v test_safe_spiral test_stain_scaling
python privacy_scan.py .
```

| File | Purpose |
|---|---|
| `train_vgms.py` | Main training and evaluation |
| `safe_spiral.py` | Bounded spatial search |
| `stain_scaling.py` | Training-fitted scalar standardization |
| `select_alignment_lambda.py` | Alignment-mixture selection |
| `run_module_ablation.py` | Input, alignment, and query-branch variants |
| `run_text_encoder_ablation.py` | Text representation comparisons |
| `run_image_encoder_ablation.py` | Image representation comparisons |
| `summarize_results.py` | Result-table export |

The main training outputs provide query attention for visualization. Additional scripts support cohort plotting and single-stain experiments.
