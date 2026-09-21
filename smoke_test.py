"""Small CPU smoke test for the paper-protocol model and split logic."""

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
    assert outputs["align_indices"].shape == (7, 1)
    assert outputs["s_feat"].shape == (7, 11)
    assert outputs["s_morph"].shape == (7, 11)
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
    model = VGMS(torch.randn(6, 24), feature_dim=16, hidden_dim=12, dropout=0.0)
    sample = make_sample(n_he=277, n_ihc=7)
    for mode in ["both", "feature", "morph"]:
        model.alignment_mode = mode
        _, output = model(sample)
        _, _, dense, _ = model._similarity(sample["he_features"], sample["ihc_features"],
                                           sample["he_h_map"], sample["ihc_h_map"])
        assert torch.allclose(output["s_align"], dense, atol=1e-6)
        assert torch.equal(output["align_indices"], dense.argmax(dim=1, keepdim=True))
    # Equal scores allow many anchors to choose candidate zero, even when
    # candidate zero is not first in their spiral traversal.
    sample["he_features"].zero_()
    sample["ihc_features"].zero_()
    model.alignment_mode = "feature"
    _, output = model(sample)
    assert torch.equal(output["align_indices"], torch.zeros(7, 1, dtype=torch.long))


def main():
    torch.manual_seed(42)
    np.random.seed(42)
    test_spiral_matching()
    test_forward_backward()
    test_patient_disjoint_split()
    print("VGMS smoke test passed.")


if __name__ == "__main__":
    main()
