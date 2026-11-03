import copy
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from routecraft.api import Batch, create_app
from routecraft.data import LABELS, fingerprint, load, validate_splits
from routecraft.engine import Engine, verify_bundle
from routecraft.models import AdapterRouter, probabilities, tokenize, tokenizer
from routecraft.train import frozen_hash, select_policy, train

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"


@pytest.fixture(scope="session")
def engine():
    return Engine(ARTIFACTS)


@pytest.fixture
def client():
    with TestClient(create_app(ARTIFACTS, api_token="")) as c:
        yield c


def test_source_split_roles_are_disjoint():
    data = load(ROOT / "data/task.json")
    assert data["labels"] == LABELS
    assert {k: len(v) for k, v in data["splits"].items()} == {
        "train": 1000,
        "selection": 100,
        "calibration": 50,
        "policy": 50,
        "policy_oos": 100,
        "test": 300,
        "test_oos": 1000,
    }
    assert {r["label"] for r in data["splits"]["train"]} == set(LABELS)
    assert all(r["label"] == "oos" for r in data["splits"]["test_oos"])


def test_normalized_cross_split_duplicate_rejected():
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits(
            {
                "train": [{"text": "Book a Flight", "label": "book_flight"}],
                "test": [{"text": "  book   a flight  ", "label": "book_flight"}],
            }
        )


def test_no_overlap_between_id_and_ood():
    data = load(ROOT / "data/task.json")
    mutated = copy.deepcopy(data["splits"])
    mutated["test_oos"][0]["text"] = mutated["train"][0]["text"]
    with pytest.raises(ValueError, match="overlaps"):
        validate_splits(mutated)


def test_probabilities_stable_for_extreme_logits():
    p = probabilities(np.array([[1e5, -1e5, 1e5], [0, 0, 0]]))
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p.sum(axis=1), 1)
    assert p[0, 1] == 0


def test_fitted_review_policy_uses_separate_arrays():
    logits = np.eye(10) * 3
    oos = np.zeros((10, 10))
    policy = select_policy(logits, np.arange(10), logits, oos)
    assert 0.4 <= policy["temperature"] <= 5
    assert 0.1 <= policy["threshold"] <= 0.99
    assert policy["policy_balanced_accuracy"] == 1


def test_loaded_bundle_checksums():
    manifest = verify_bundle(ARTIFACTS)
    assert len(manifest["bundle_version"]) == 16
    assert "adapter/adapter_model.safetensors" in manifest["files"]


def test_tampered_head_rejected(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(ARTIFACTS, copied)
    (copied / "head.json").write_text("{}")
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_bundle(copied)


def test_manifest_cannot_leave_bundle(tmp_path):
    copied = tmp_path / "bundle"
    shutil.copytree(ARTIFACTS, copied)
    p = copied / "manifest.json"
    manifest = json.loads(p.read_text())
    manifest["files"]["../outside.txt"] = "irrelevant"
    manifest["bundle_version"] = fingerprint(manifest["files"])[:16]
    p.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leaves the bundle"):
        verify_bundle(copied)


def test_lora_step_updates_adapter_but_not_pretrained_weights():
    torch.manual_seed(123)
    model = AdapterRouter(ARTIFACTS / "base")
    base_before = frozen_hash(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters() if "lora_" in n}
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    assert all("lora_" in n or n.startswith("head.") for n in names)
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 9482
    encoded = tokenize(tokenizer(ARTIFACTS / "base"), ["book me a flight", "check my order"])
    loss = torch.nn.functional.cross_entropy(model(**encoded), torch.tensor([0, 4]))
    loss.backward()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    optimizer.step()
    assert frozen_hash(model) == base_before
    assert any(not torch.equal(before[n], p) for n, p in model.named_parameters() if n in before)


def test_adapter_decisions_and_baselines(engine):
    result = engine.route(
        ["Can you check where my order is?", "Explain the theory of quantum gravity."]
    )
    assert result["results"][0]["models"]["lora"]["predicted_intent"] == "order_status"
    assert result["results"][0]["models"]["lora"]["decision"] == "route"
    assert result["results"][1]["models"]["lora"]["decision"] == "review"
    for row in result["results"]:
        assert set(row["models"]) == {"lora", "lexical", "frozen"}
        for model in row["models"].values():
            values = [r["probability"] for r in model["top_intents"]]
            assert values == sorted(values, reverse=True)
            assert 0 <= model["confidence"] <= 1


def test_single_and_batched_logits_agree(engine):
    text = "Can you check where my order is?"
    one = engine.route([text])["results"][0]
    two = engine.route([text, text])["results"][0]
    for name in one["models"]:
        assert one["models"][name]["confidence"] == pytest.approx(
            two["models"][name]["confidence"], abs=1e-5
        )


def test_token_limit_is_visible(engine):
    row = engine.route(["weather " * 100])["results"][0]
    assert row["truncated"] is True
    assert row["tokens"] > 48


@pytest.mark.parametrize("value", [[], [""], [" "], ["a"], ["x" * 1001], ["ok"] * 17, [False]])
def test_input_contract_rejects_invalid_batches(value):
    with pytest.raises(ValueError):
        Batch(texts=value)


def test_api_contract_and_metrics(client):
    assert client.get("/health").status_code == 200
    assert client.get("/ready").status_code == 200
    assert client.get("/").status_code == 200
    assert client.get("/api/config").json()["language"] == "English"
    result = client.post("/api/route", json={"texts": ["Can you check where my order is?"]})
    assert result.status_code == 200
    assert result.json()["request_id"]
    assert result.json()["results"][0]["models"]["lora"]["predicted_intent"] == "order_status"
    assert "routecraft_texts_total 1" in client.get("/metrics").text
    assert (
        client.post(
            "/api/route", json={"texts": ["hello"], "provider_url": "https://example.com"}
        ).status_code
        == 422
    )


def test_token_protects_read_and_write():
    with TestClient(create_app(ARTIFACTS, api_token="secret")) as c:
        for path in ["/api/config", "/api/report", "/metrics"]:
            assert c.get(path).status_code == 401
            assert c.get(path, headers={"Authorization": "Bearer secret"}).status_code == 200
        assert c.post("/api/route", json={"texts": ["hello"]}).status_code == 401


def test_missing_artifacts_fail_readiness(tmp_path):
    with TestClient(create_app(tmp_path)) as c:
        assert c.get("/health").status_code == 200
        assert c.get("/ready").status_code == 503
        assert c.get("/api/report").status_code == 503


def test_release_report_matches_policy_and_manifest():
    report = json.loads((ARTIFACTS / "report.json").read_text())
    assert report["frozen_parameters_unchanged"] is True
    assert report["trainable_parameters"] == 9482
    assert report["policies"] == json.loads((ARTIFACTS / "policies.json").read_text())
    assert 1 <= report["selected_epoch"] <= len(report["history"])
    for result in report["test"].values():
        assert sum(sum(row) for row in result["confusion_matrix"]) == 300
        assert 0 <= result["routing"]["out_of_scope_false_accept_rate"] <= 1


def test_changing_test_labels_cannot_change_trained_parameters_or_policy(tmp_path):
    data = load(ROOT / "data/task.json")
    changed = copy.deepcopy(data)
    for row in changed["splits"]["test"]:
        row["label"] = LABELS[(LABELS.index(row["label"]) + 1) % len(LABELS)]
    changed["dataset_sha256"] = fingerprint(changed["splits"])
    source = tmp_path / "original.json"
    source.write_text(json.dumps(data))
    alternate = tmp_path / "changed.json"
    alternate.write_text(json.dumps(changed))
    left, right = tmp_path / "left", tmp_path / "right"
    for target in [left, right]:
        shutil.copytree(ARTIFACTS / "base", target / "base")
    train(source, left, epochs=1, seed=13)
    train(alternate, right, epochs=1, seed=13)
    for name in [
        "adapter/adapter_model.safetensors",
        "head.json",
        "policies.json",
        "frozen.json",
        "lexical.json",
    ]:
        assert (left / name).read_bytes() == (right / name).read_bytes()
