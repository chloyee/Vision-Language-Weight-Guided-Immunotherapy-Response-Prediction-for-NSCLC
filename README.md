# VGMS

Reference implementation for:

**VISION-LANGUAGE WEIGHT-GUIDED IMMUNOTHERAPY RESPONSE PREDICTION FOR NSCLC USING MULTI-STAIN WHOLE-SLIDE IMAGES**

VGMS integrates paired hematoxylin and eosin (H&E) and PD-L1 immunohistochemistry (IHC) whole-slide image representations for pathological complete response prediction in non-small cell lung cancer. The repository contains the main training pipeline, single-stain baselines, module and encoder ablations, cohort visualization, result summarization, and query-attention heatmap generation.

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
  --val_size 0.2 \
  --epochs 200 \
  --patience 15 \
  --device cuda:0 \
  --amp --tf32
```

Checkpoint selection and early stopping use only the validation subset derived from the training split. Internal and external test cohorts are excluded from training, hyperparameter selection, and early stopping.

## Ablations

```bash
python run_module_ablation.py --base_path ./data/patches --device cuda:0
python run_text_encoder_ablation.py --base_path ./data/patches --device cuda:0
python run_image_encoder_ablation.py --base_path ./data/patches --device cuda:0
```

Use `--dry_run` to inspect generated commands before launching experiments.

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

## Citation

Citation information will be added after publication.

## License

This private review repository is provided for evaluation only. No redistribution or reuse license is granted at this stage.
