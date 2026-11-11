"""Compare reusable calculations with the frozen manuscript evidence."""

import json

import numpy as np
import pandas as pd
from dcap.analysis import replay
from dcap.evaluation import METRICS
from dcap.paths import ROOT, read


def test_published_metrics_intervals_and_grid(tmp_path):
    computed = replay(tmp_path, grid=True)
    expected = read(ROOT / "results/representative.json")
    for key, r in computed.items():
        for metric in METRICS:
            np.testing.assert_allclose(
                r["outcomes"][metric], expected[key][metric], rtol=0, atol=1e-14
            )
        if key.startswith("dcap"):
            assert r["preservation"]["joint_support"] == expected[key]["joint_support"]
    reference = pd.read_csv(ROOT / "results/figure7.csv")
    for row in reference[reference.method == "dcap"].itertuples():
        r = computed["dcap70_" + row.split]["preservation"]["metrics"][row.metric]
        np.testing.assert_allclose(
            [r["delta"], *r["ci95"]],
            [row.delta, row.ci_low, row.ci_high],
            rtol=0,
            atol=1e-14,
        )
    keys = [
        "split",
        "width_km",
        "origin_x_fraction",
        "origin_y_fraction",
        "margin_scale",
    ]
    actual = (
        pd.read_csv(tmp_path / "figure8.csv").sort_values(keys).reset_index(drop=True)
    )
    frozen = (
        pd.read_csv(ROOT / "results/figure8.csv")
        .sort_values(keys)
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(actual[frozen.columns], frozen, check_exact=True)


def test_split_and_calibration_provenance():
    design = read(ROOT / "configs/design.json")
    frame = pd.read_csv(ROOT / "metadata/split.csv")
    for key, split in [
        ("fit_ids", "train"),
        ("validation_ids", "validation"),
        ("outer_ids", "test"),
    ]:
        assert set(design[key]) == set(frame[frame.split == split].sample_id)
    for key in ["gate_ids", "taylor_ids", "inheritance_ids"]:
        assert set(design[key]) <= set(design["fit_ids"])
    assert frame.groupby("spatial_group").split.nunique().max() == 1
    assert design["normalization"] == read(ROOT / "metadata/normalization.json")


def test_reject_thresholds_from_other_checkpoints(tmp_path, monkeypatch):
    import dcap.models as models
    from dcap.evaluation import predict_ensemble

    paths = []
    for n in range(3):
        p = tmp_path / f"m{n}.pt"
        p.write_bytes(str(n).encode())
        paths.append(p)
    monkeypatch.setattr(
        models,
        "load_checkpoint",
        lambda path, device: (
            None,
            {"configuration": {"input_mode": "reflectance_change"}},
        ),
    )
    wrong = tmp_path / "validation.json"
    wrong.write_text(
        json.dumps(
            {
                "split": "validation",
                "checkpoint_hashes": ["wrong"] * 3,
                "precision": "fp32",
            }
        )
    )
    import pytest

    with pytest.raises(ValueError, match="do not belong"):
        predict_ensemble(
            "custom",
            "test",
            device="cpu",
            precision="fp32",
            checkpoints=paths,
            validation_result=wrong,
        )
    with pytest.raises(ValueError, match="requires --validation-result"):
        predict_ensemble(
            "custom", "test", device="cpu", precision="fp32", checkpoints=paths
        )


def test_inventory_quality_lineage():
    import hashlib

    from dcap.preprocess import metadata

    data = metadata("source_inventory.geojson.gz")
    quality = metadata("source_quality.json.gz")
    ids = {f["properties"]["polygon_id"] for f in data["features"]}
    assert len(ids) == 2063 and ids == set(quality)
    assert (
        hashlib.sha256(("\n".join(sorted(ids)) + "\n").encode()).hexdigest()
        == "16e223cba24303b7e306311143aa76564e9230f988655a9d1c1ccf851b80ae2c"
    )
    assert all(
        float(v[k]) <= 0.2
        for q in quality.values()
        for v in q.values()
        for k in ["context_bad_fraction", "polygon_bad_fraction"]
    )


def test_campaign_plan():
    from dcap.experiment import campaign

    p = campaign()
    assert p["jobs"] == 63
    dense = [j for j in p["training"] if j["method"] == "dense"]
    assert len(dense) == 3
    for method in ["dcap", "taylor", "magnitude"]:
        selected = [j for j in p["training"] if j["method"] == method]
        assert len(selected) == 15 and all(
            any(j["recovery_weights"].values()) for j in selected
        )


def test_physically_pruned_checkpoint_roundtrip(tmp_path):
    import torch
    from dcap.models import build_model, load_checkpoint, save_checkpoint
    from dcap.structural import prune_to_measured_mac_target

    torch.set_num_threads(2)
    torch.manual_seed(19)
    configuration = {
        "architecture": "resunet",
        "input_channels": 21,
        "input_mode": "reflectance_change",
    }
    model = build_model(configuration).eval()
    x = torch.randn(1, 21, 32, 32)
    pruned, measurement, _ = prune_to_measured_mac_target(
        model,
        x,
        method="group_magnitude_l2",
        target_mac_reduction=0.25,
        iterative_steps=25,
        achieved_mac_tolerance=0.08,
    )
    assert (
        measurement.within_tolerance
        and measurement.pruned_parameters < measurement.baseline_parameters
    )
    assert pruned(x).shape == (1, 1, 32, 32)
    path = tmp_path / "pruned.pt"
    save_checkpoint(pruned, path, configuration)
    restored, _ = load_checkpoint(path)
    with torch.inference_mode():
        torch.testing.assert_close(pruned(x), restored(x), rtol=0, atol=0)


def test_validation_selection_includes_ap_guard():
    from dcap.experiment import select_operating_points

    rows = pd.read_csv(ROOT / "results/selection.csv").to_dict("records")
    dense = read(ROOT / "results/representative.json")["dense_validation"][
        "average_precision"
    ]
    selected = select_operating_points(rows, dense)
    assert selected["dcap"] == "E1_original_dcap_mac70_full"
    assert selected["taylor"] == "E1_original_taylor_mac85_full"
    assert selected["magnitude"] == "dense"
    assert selected["overall_included_pruning_methods"] == selected["taylor"]
    # Perfect support alone must not admit a model whose AP violates the guard.
    bad = dict(
        ensemble="bad",
        method="dcap",
        mac_reduction=99.0,
        validation_ap=dense - 0.011,
        joint_support=1.0,
    )
    assert select_operating_points([bad], dense)["dcap"] == "dense"


def test_evaluation_rejects_changed_statistics(tmp_path):
    import pytest
    from dcap.evaluation import SCALARS, load_assessment, save_assessment

    path = tmp_path / "evaluation.json"
    save_assessment(
        path, {"split": "test"}, ["chip1"], np.zeros((1, len(SCALARS) + 2002))
    )
    record, ids, values = load_assessment(path)
    assert record["split"] == "test" and ids.tolist() == ["chip1"]
    path.with_suffix(".npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed evaluation statistics"):
        load_assessment(path)


def test_campaign_requires_complete_unchanged_checkpoint(tmp_path):
    import pytest
    from dcap.experiment import completed_checkpoint
    from dcap.paths import sha256, write

    with pytest.raises(ValueError, match="Incomplete training"):
        completed_checkpoint(tmp_path)
    checkpoint = tmp_path / "checkpoint.pt"
    structure = tmp_path / "structure.json"
    checkpoint.write_bytes(b"test checkpoint")
    structure.write_text("{}")
    write(
        tmp_path / "complete.json",
        {
            "interrupted": False,
            "smoke_only": False,
            "checkpoint_sha256": sha256(checkpoint),
            "structure_sha256": sha256(structure),
        },
    )
    assert completed_checkpoint(tmp_path) == checkpoint
    checkpoint.write_bytes(b"different checkpoint")
    with pytest.raises(ValueError, match="changed completed output"):
        completed_checkpoint(tmp_path)


def test_cli_rejects_invalid_epochs_and_explains_commands():
    import pytest
    from dcap.cli import build_parser

    parser = build_parser()
    assert "spatial evidence" in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["train", "--epochs", "0"])
    args = parser.parse_args(
        ["evaluate", "--split", "test", "--device", "cpu", "--precision", "fp32"]
    )
    assert args.device == "cpu"


def test_package_excludes_caches_downloads_and_logs(tmp_path):
    import importlib.util

    module_path = ROOT / "tools/package_release.py"
    spec = importlib.util.spec_from_file_location("release_packaging", module_path)
    packaging = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(packaging)
    packaging.ROOT = tmp_path
    for name in [
        "src/dcap/core.py",
        "src/dcap/__pycache__/core.pyc",
        "src/dcap/debug.log",
        "downloads/data/chips.zip",
        "generated/result.json",
        "data/images/chip.tif",
    ]:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        assert packaging.included(path) == (name == "src/dcap/core.py")


def test_gate_bank_preserves_pytorch_module_api():
    import torch
    from dcap.algorithm import SharedGateBank
    from dcap.models import build_model

    torch.set_num_threads(2)
    model = build_model({"architecture": "resunet", "input_channels": 21}).eval()
    bank = SharedGateBank(model, torch.zeros(1, 21, 32, 32))
    assert list(bank.modules())[0] is bank
    assert {id(p) for p in bank.parameters()}.isdisjoint(
        id(p) for p in model.parameters()
    )
    bank.mac_fraction().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in bank.parameters()
    )


def test_target_device_tables_from_recorded_timings():
    from dcap.figures import deployment_results

    table, figure = deployment_results(ROOT / "results")
    for actual, filename in [(table, "table13.csv"), (figure, "figure10.csv")]:
        expected = pd.read_csv(ROOT / "results" / filename).fillna("")
        pd.testing.assert_frame_equal(actual, expected, check_dtype=False, rtol=1e-13, atol=1e-13)
    assert set(table.model) == {"dense", "dcap70", "taylor70", "sirfp70", "segformer_b2"}
    assert len(table) == 18 and len(figure) == 20
