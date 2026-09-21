# VGMS

Reference implementation for:

**VISION-LANGUAGE WEIGHT-GUIDED IMMUNOTHERAPY RESPONSE PREDICTION FOR NSCLC USING MULTI-STAIN WHOLE-SLIDE IMAGES**

VGMS integrates paired hematoxylin and eosin (H&E) and PD-L1 immunohistochemistry (IHC) whole-slide image representations for pathological complete response prediction in non-small cell lung cancer. The repository contains the main training pipeline, single-stain baselines, module and encoder ablations, cohort visualization, result summarization, and query-attention heatmap generation.

The default configuration follows the reported protocol. Each sampled IHC patch is matched to the highest-scoring H&E candidate from the same case using a fixed validation-selected mixture of hematoxylin-channel SSIM and CHIEF feature cosine similarity. Matched H&E and IHC CHIEF embeddings are concatenated with H&E hematoxylin and IHC DAB intensity features and projected. Six frozen ClinicalBERT query embeddings attend to the aligned visual tokens, and the six query-specific evidence vectors are concatenated for a three-layer gated MLP classifier.

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

The study uses retrospectively collected data from two hospitals. All splits are patient-disjoint. No image, clinical, label, split, or model-output files are included in this repository.

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

Additional cohort characteristics, eligibility criteria, preprocessing details, and response distributions are reported in the paper.

## Data availability

The source WSIs and associated clinical annotations cannot be made publicly available because they contain protected patient information and are governed by institutional ethics approvals and data-use restrictions. Access is therefore not provided through this repository. Researchers interested in validation should contact the corresponding authors and obtain separate institutional and ethics approval; availability remains subject to the participating hospitals' policies.

The code expects de-identified identifiers. Do not place raw clinical exports, patient names, medical-record numbers, original split files, or identifiable slide filenames under version control.

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
  --epochs 200 \
  --patience 15 \
  --loss focal \
  --morphology_mode macenko \
  --alignment_lambda 0.5 \
  --max_patches_per_case 1024 \
  --device cuda:0 \
  --amp --tf32
```

Checkpoint selection and early stopping use only the validation subset derived from the training split. Internal and external test cohorts are excluded from training, hyperparameter selection, and early stopping.

### Paper protocol defaults

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
| Alignment | Hard top-1 match; fixed `alignment_lambda` selected on validation AUC |
| Optimizer | AdamW, learning rate `1e-4`, weight decay `1e-4` |
| Schedule | Cosine decay to `1e-6`, at most 200 epochs |
| Loss | Binary focal loss, gamma 2, training-label-derived class weights |
| Early stopping | Validation AUC, patience 15 |
| Decision threshold | 0.5 |
| AUC confidence interval | 1,000 pair-level bootstrap resamples, percentile limits, seed 42 |

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

Matching starts at each IHC anchor's corresponding location in the shared, already scale-mapped coordinate frame. Tile names are `y_x.png` (vertical then horizontal coordinate), with matching feature stems `y_x.pt`. Candidates are visited in clockwise expanding square rings, starting right and proceeding down, left, and up. Empty grid locations are skipped. Similarity is evaluated in batches of at most 128 retained candidates in spiral order. Every retained H&E candidate is visited without stopping at local maxima. The highest score wins, with ties resolved by the lowest original candidate index. Many IHC anchors may select the same H&E patch. Existing caches remain usable because coordinates are parsed from retained tile names. Invalid coordinate names raise an error.

## Ablations

```bash
python run_module_ablation.py --base_path ./data/patches --alignment_lambda 0.5 --device cuda:0
python run_text_encoder_ablation.py --base_path ./data/patches --alignment_lambda 0.5 --device cuda:0
python run_image_encoder_ablation.py --base_path ./data/patches --alignment_lambda 0.5 --device cuda:0
```

Replace `0.5` with the value selected by `select_alignment_lambda.py`. Use `--dry_run` to inspect generated commands before launching experiments.

When vision-language weighting is disabled, the ablation uses a learned gated-attention MIL pooler instead of uniform averaging. This keeps the no-query comparison trainable while leaving the full VGMS query-attention path unchanged.

## Query-attention visualization

Create a private slide manifest following `slides.example.csv`, then run:

```bash
python visualize_query_attention.py \
  --slides_csv ./slides.example.csv \
  --ctranspath_weight ./model_weight/CHIEF_CTransPath.pth \
  --out_dir ./results/query_attention \
  --tile_size 80 --batch_size 64 --amp
```

Do not commit a populated manifest containing real slide paths or identifiers.

## Reproducibility and privacy

- Random seed defaults to 42.
- Test cohorts are evaluation-only.
- Generated predictions retain de-identified case and slide keys for paired statistical analysis.
- `data/`, model weights, checkpoints, caches, predictions, and WSI formats are excluded by `.gitignore`.
- Run `python privacy_scan.py .` before every release commit.
- Run `python verify_paper_protocol.py` to check paper-protocol defaults and core implementation markers.
- Run `python smoke_test.py` after installing dependencies to validate model shapes, gradients, and patient-disjoint splitting.

## Citation

Citation information will be added after publication.

## License

This repository is provided for research evaluation. A formal reuse license will be added with the public release; until then, no redistribution or reuse license is granted.
