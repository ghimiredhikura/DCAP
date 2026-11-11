"""Comparator configuration, initialization and portable-checkpoint contracts."""

import os

import numpy as np
import pytest
import torch

from dcap.paths import ROOT, read


def test_comparator_catalog_and_thresholds():
    catalog = read(ROOT / "metadata/models.json")
    thresholds = read(ROOT / "metadata/thresholds.json")
    expected = read(ROOT / "results/comparators.json")
    from dcap.assets import records_for

    assets = {r["asset"]: r for r in catalog["models"]}
    for name in ("sirfp25", "sirfp70", "segformer_b2"):
        records = [assets[a] for a in catalog["ensembles"][name]["assets"]]
        assert [r["seed"] for r in records] == [42, 123, 2026]
        assert all(r["release"] == "v1.2.0" for r in records)
        assert len(thresholds[name]["cross_fitted_thresholds"]) == 49
        for split in ("validation", "test"):
            assert (
                expected[f"{name}_{split}"]["global_validation_threshold"]
                == thresholds[name]["global_validation_threshold"]
            )
    main = records_for("models", "main")
    comparators = records_for("models", "comparators")
    assert len(main) == 7 and all(r["release"] == "v1.1.0" for r in main)
    assert len(comparators) == 11
    assert {r["asset"] for r in comparators if not r["asset"].endswith(".pt")} == {
        "COMPARATOR_MODEL_NOTICES.txt",
        "SegFormer-LICENSE.txt",
    }


def test_comparator_training_protocol():
    from dcap.training import job_for

    job = job_for("sirfp", 123, 0.25)
    assert job["name"] == "sirfp_mac25_seed123"
    assert all(v > 0 for v in job["recovery_weights"].values())
    job = job_for("dense", 42, architecture="segformer_b2")
    assert job["training_settings"] == dict(
        encoder_lr=6e-5, decoder_lr=6e-4, weight_decay=0.01
    )
    assert not any(job["recovery_weights"].values())
    with pytest.raises(ValueError, match="uses --method dense"):
        job_for("dcap", architecture="segformer_b2")


def test_segformer_tensor_checkpoint_and_optimizer(tmp_path):
    pytest.importorskip("timm")
    from dcap.models import build_model, load_checkpoint, save_checkpoint

    torch.set_num_threads(2)
    model = build_model(dict(architecture="segformer_b2", input_channels=21)).eval()
    assert sum(p.numel() for p in model.parameters()) == 27403841
    stem = model.encoder.patch_embed1.proj.weight
    other = [i for i in range(21) if i not in (8, 7, 6)]
    assert torch.count_nonzero(stem[:, other]) == 0
    groups = model.optimizer_parameter_groups(6e-5, 6e-4)
    parameters = [id(p) for group in groups for p in group["params"]]
    assert len(parameters) == len(set(parameters)) == len(list(model.parameters()))
    norm_ids = {
        id(p)
        for m in model.modules()
        if isinstance(m, (torch.nn.LayerNorm, torch.nn.BatchNorm2d))
        for p in m.parameters()
    }
    assert {
        id(p) for g in groups if g["weight_decay"] == 0 for p in g["params"]
    } == norm_ids
    path = tmp_path / "segformer.pt"
    save_checkpoint(model, path, dict(architecture="segformer_b2", input_channels=21))
    restored, _ = load_checkpoint(path)
    sample = torch.randn(1, 21, 32, 32)
    with torch.inference_mode():
        torch.testing.assert_close(restored(sample), model(sample), atol=0, rtol=0)
    assert model.attention_matmul_macs(256, 256) == 243269632


def test_sirfp_rejects_unpinned_source(tmp_path):
    from dcap.sirfp import load_scorer

    (tmp_path / "pruners").mkdir()
    (tmp_path / "pruners/graph_pruner.py").write_text(
        "raise RuntimeError('must never execute')"
    )
    with pytest.raises(ValueError, match="differs from the pinned"):
        load_scorer(tmp_path)


def test_sirfp_upstream_score_fixture():
    source = os.environ.get("DCAP_SIRFP_SOURCE")
    if not source:
        pytest.skip("Set DCAP_SIRFP_SOURCE to the separately obtained pinned checkout")
    from dcap.sirfp import load_scorer

    scorer = load_scorer(source)
    gram = torch.tensor([[0.7, 0.1, 0.2], [0.1, 0.5, 0.1], [0.2, 0.1, 0.4]])
    edge = torch.tensor([0.3, 0.2, 0.1])
    np.testing.assert_allclose(scorer(gram, edge, 0.0), [-1.0, -0.6, -0.4], atol=1e-7)
    np.testing.assert_allclose(
        scorer(gram, edge, 0.4), [-0.88, -0.52, -0.36], atol=1e-7
    )
