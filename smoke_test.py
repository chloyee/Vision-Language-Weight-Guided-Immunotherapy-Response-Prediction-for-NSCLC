"""CPU checks for VGMS training, alignment and patient partitions."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path
from PIL import Image

import numpy as np
import pandas as pd
import torch

from train_vgms import (
    DEFAULT_QUERIES,
    FocalLoss,
    VGMS,
    stratified_split_cases,
    spiral_candidate_order,
    tile_coordinate_yx,
)


def make_sample(feature_dim=16, n_he=11, n_ihc=7, map_size=8):
    generator = torch.Generator().manual_seed(42)
    return {
        "he_features": torch.randn(n_he, feature_dim, generator=generator),
        "ihc_features": torch.randn(n_ihc, feature_dim, generator=generator),
        "he_h": torch.rand(n_he, generator=generator),
        "ihc_h": torch.rand(n_ihc, generator=generator),
        "ihc_dab": torch.rand(n_ihc, generator=generator),
        "he_morph": torch.randn(n_he, 5, generator=generator),
        "ihc_morph": torch.randn(n_ihc, 5, generator=generator),
        "he_h_map": torch.rand(n_he, map_size, map_size, generator=generator),
        "ihc_h_map": torch.rand(n_ihc, map_size, map_size, generator=generator),
        "ihc_source_indices": torch.arange(n_ihc),
        "he_candidate_tile_names": [f"{i // 4}_{i % 4}" for i in range(n_he)],
        "ihc_anchor_tile_names": [f"{i // 4}_{i % 4}" for i in range(n_ihc)],
    }


def test_forward_backward():
    query_embeddings = torch.randn(len(DEFAULT_QUERIES), 24)
    model = VGMS(
        query_embeddings=query_embeddings,
        feature_dim=16,
        hidden_dim=12,
        align_dim=8,
        dropout=0.0,
        alignment_lambda=0.5,
    )
    sample = make_sample()
    logit, outputs = model(sample)
    assert logit.ndim == 0
    assert outputs["query_weights"].shape == (6, 7)
    assert outputs["align_indices"].shape == (7, 11)
    assert outputs["align_weights"].shape == (7, 11)
    assert outputs["s_feat"].shape == (7, 11)
    assert outputs["s_morph"].shape == (7, 11)
    assert outputs["s_base"].shape == (7, 11)
    assert outputs["pairwise_distance"].shape == (7, 11)
    assert torch.allclose(
        outputs["align_weights"].sum(dim=1),
        torch.ones(7),
        atol=1e-6,
    )
    assert torch.allclose(
        outputs["query_weights"].sum(dim=1),
        torch.ones(6),
        atol=1e-5,
    )

    loss = FocalLoss(gamma=2.0, alpha_pos=0.6, alpha_neg=0.4)(
        logit.view(1), torch.ones(1)
    )
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())

    no_query_model = VGMS(
        query_embeddings=query_embeddings,
        feature_dim=16,
        hidden_dim=12,
        align_dim=8,
        dropout=0.0,
        disable_query_branch=True,
    )
    _, no_query_outputs = no_query_model(make_sample())
    assert torch.allclose(
        no_query_outputs["mil_weights"].sum(),
        torch.tensor(1.0),
        atol=1e-5,
    )
    assert torch.allclose(
        no_query_outputs["query_weights"][0],
        no_query_outputs["mil_weights"],
        atol=1e-6,
    )


def test_patient_disjoint_split():
    rows = []
    for patient_index in range(20):
        for pair_index in range(1 + patient_index % 2):
            rows.append(
                {
                    "case_id": f"case_{patient_index:03d}",
                    "pair_case_name": f"pair_{patient_index:03d}_{pair_index}",
                    "label": patient_index % 2,
                }
            )
    frame = pd.DataFrame(rows)
    train, val, test = stratified_split_cases(
        frame,
        test_size=0.3,
        val_size=2.0 / 7.0,
        seed=42,
    )
    groups = [set(split["case_id"]) for split in [train, val, test]]
    assert not (groups[0] & groups[1])
    assert not (groups[0] & groups[2])
    assert not (groups[1] & groups[2])
    assert [len(group) for group in groups] == [10, 4, 6]


def test_spiral_matching():
    points = [(0, 0), (0, 1), (1, 1), (1, 0), (1, -1),
              (0, -1), (-1, -1), (-1, 0), (-1, 1), (-1, 2)]
    assert spiral_candidate_order(points, (0, 0)) == list(range(len(points)))
    assert tile_coordinate_yx("250_500.png") == (250, 500)
    sparse = [points[i] for i in [8, 0, 5, 2]]
    assert spiral_candidate_order(sparse, (0, 0)) == [1, 3, 2, 0]

    model = VGMS(
        torch.randn(6, 24),
        feature_dim=16,
        hidden_dim=12,
        dropout=0.0,
        align_topk=64,
        temperature=1.0,
        spatial_gamma=0.25,
        spatial_sigma=8.0,
        patch_stride=250.0,
    )
    sample = make_sample(n_he=277, n_ihc=7)
    for mode in ["both", "feature", "morph"]:
        model.alignment_mode = mode
        _, output = model(sample)
        _, _, base, _ = model._similarity(
            sample["he_features"],
            sample["ihc_features"],
            sample["he_h_map"],
            sample["ihc_h_map"],
        )
        expected = base - model.spatial_gamma * output["spatial_penalty"]
        expected_indices = torch.argsort(
            expected, dim=1, descending=True, stable=True
        )[:, :64]
        assert torch.allclose(output["s_base"], base, atol=1e-6)
        assert torch.allclose(output["s_align"], expected, atol=1e-6)
        assert torch.equal(output["align_indices"], expected_indices)
        assert output["align_indices"].shape == (7, 64)
        assert torch.allclose(
            output["align_weights"].sum(dim=1), torch.ones(7), atol=1e-6
        )

    # With equal feature scores, the spatial term makes the exact-coordinate
    # H&E patch the highest-ranked candidate for each IHC anchor.
    sample["he_features"].zero_()
    sample["ihc_features"].zero_()
    model.alignment_mode = "feature"
    _, output = model(sample)
    assert torch.equal(output["align_indices"][:, 0], torch.arange(7))


def test_training_pipeline():
    """Run the actual entry point on synthetic RGB patches and frozen features."""
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        cohort = root / "data" / "cohort1"
        rng = np.random.default_rng(42)
        rows = []
        for i in range(20):
            he, ihc = f"he_{i:03d}", f"ihc_{i:03d}"
            pair = he + "__" + ihc
            feature_dir = cohort / "FusionFeatures_patch_level_tiles" / pair
            he_dir = cohort / "patches_lv0_ps250" / "cohort1_HE_slides_data" / he
            ihc_dir = cohort / "patches_lv0_ps250" / "cohort1_IHC_slides_data" / ihc
            for path in (feature_dir, he_dir, ihc_dir):
                path.mkdir(parents=True, exist_ok=True)
            for tile in ("0_0", "0_250"):
                torch.save(torch.from_numpy(rng.normal(size=(2, 16)).astype(np.float32)), feature_dir / (tile + ".pt"))
                for path in (he_dir, ihc_dir):
                    Image.fromarray(rng.integers(30, 230, size=(250, 250, 3), dtype=np.uint8)).save(path / (tile + ".png"))
            rows.append(dict(case_id=f"patient_{i:03d}", he_file=he + ".svs", ihc_file=ihc + ".svs", pair_case_name=pair, label=i % 2))
        pd.DataFrame(rows).to_csv(cohort / "pair_labels_with_case_id.csv", index=False)
        query_file = root / "queries.npy"
        np.save(query_file, rng.normal(size=(6, 24)).astype(np.float32))
        result = root / "results"
        subprocess.run([
            sys.executable, str(Path(__file__).with_name("train_vgms.py")),
            "--base_path", str(root / "data"), "--external_project_name", "",
            "--results_dir", str(result), "--query_embedding_path", str(query_file),
            "--epochs", "1", "--patience", "1", "--num_workers", "0", "--device", "cpu",
            "--feature_dim", "16", "--hidden_dim", "12", "--align_topk", "2",
            "--n_bootstraps", "20", "--no_save_train_intermediates",
        ], check=True)
        checkpoint = torch.load(result / "best_model.pth", map_location="cpu")
        assert checkpoint["args"]["stain_scalar_mode"] == "train_standardized"
        with open(result / "stain_scaler.json") as stream:
            scaler = json.load(stream)
        assert checkpoint["args"]["stain_scaler"] == scaler
        assert scaler["channels"]["he_h"]["count"] == 20
        predictions = pd.read_csv(result / "internal_test_best_wsi_predictions.csv")
        assert len(predictions) == 6
        assert predictions["predict_probability"].between(0, 1).all()
        groups = [set(pd.read_csv(result / "splits" / name)["case_id"]) for name in ("train_cases.csv", "val_cases.csv", "internal_test_cases.csv")]
        assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    test_spiral_matching()
    test_forward_backward()
    test_patient_disjoint_split()
    test_training_pipeline()
    print("VGMS smoke test passed.")


if __name__ == "__main__":
    main()
