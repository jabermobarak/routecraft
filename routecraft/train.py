"""Train on source training rows; isolate checkpoint, calibration, policy and final test."""

from __future__ import annotations

import copy
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, log_loss

from .data import BASE_ID, BASE_REVISION, LABELS, MAX_LENGTH, fingerprint, load
from .models import (
    AdapterRouter,
    embeddings,
    infer,
    linear_logits,
    probabilities,
    tokenize,
    tokenizer,
)


def labels(rows):
    return np.array([LABELS.index(r["label"]) for r in rows])


def classification(logits, y):
    p = probabilities(logits)
    predicted = p.argmax(axis=1)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_f1": float(
            f1_score(y, predicted, labels=range(len(LABELS)), average="macro", zero_division=0)
        ),
        "nll": float(log_loss(y, p, labels=range(len(LABELS)))),
        "confusion_matrix": confusion_matrix(y, predicted, labels=range(len(LABELS))).tolist(),
    }


def select_policy(calibration_logits, calibration_y, policy_in, policy_oos):
    grid = np.geomspace(0.4, 5.0, 60)
    temperature = float(
        min(
            grid,
            key=lambda t: log_loss(
                calibration_y, probabilities(calibration_logits, t), labels=range(len(LABELS))
            ),
        )
    )
    confidence_in = probabilities(policy_in, temperature).max(axis=1)
    confidence_out = probabilities(policy_oos, temperature).max(axis=1)
    # Balanced ID acceptance / OOD rejection: prevalence-independent descriptive objective.
    thresholds = np.linspace(0.1, 0.99, 90)
    objectives = [
        (0.5 * np.mean(confidence_in >= t) + 0.5 * np.mean(confidence_out < t)) for t in thresholds
    ]
    index = int(np.argmax(objectives))
    return {
        "temperature": temperature,
        "threshold": float(thresholds[index]),
        "policy_balanced_accuracy": float(objectives[index]),
        "objective": "0.5 * in-scope acceptance + 0.5 * out-of-scope rejection",
    }


def routing_metrics(in_logits, y, out_logits, policy):
    p = probabilities(in_logits, policy["temperature"])
    accepted = p.max(axis=1) >= policy["threshold"]
    out_accepted = (
        probabilities(out_logits, policy["temperature"]).max(axis=1) >= policy["threshold"]
    )
    return {
        "in_scope_coverage": float(accepted.mean()),
        "accepted_in_scope_accuracy": float((p.argmax(axis=1)[accepted] == y[accepted]).mean())
        if accepted.any()
        else None,
        "out_of_scope_false_accept_rate": float(out_accepted.mean()),
        "accepted_in_scope_rows": int(accepted.sum()),
        "in_scope_rows": len(y),
        "out_of_scope_rows": len(out_logits),
        "calibrated_in_scope_nll": float(log_loss(y, p, labels=range(len(LABELS)))),
    }


def frozen_hash(model):
    h = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            h.update(name.encode())
            h.update(parameter.detach().numpy().tobytes())
    return h.hexdigest()


def write_manifest(root):
    paths = [p for p in root.rglob("*") if p.is_file() and p.name != "manifest.json"]
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    manifest = {"schema_version": 1, "files": hashes, "bundle_version": fingerprint(hashes)[:16]}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def train(dataset_path: Path, root: Path, epochs=24, seed=42):
    if epochs < 1:
        raise ValueError("epochs must be positive")
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    np.random.seed(seed)
    started = time.perf_counter()
    data = load(dataset_path)
    parts = data["splits"]
    tok = tokenizer(root / "base")
    model = AdapterRouter(root / "base")
    original_frozen = frozen_hash(model)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    texts = {k: [r["text"] for r in v] for k, v in parts.items()}
    y_train, y_select = labels(parts["train"]), labels(parts["selection"])
    encoded = tokenize(tok, texts["train"])
    target = torch.tensor(y_train)
    adapter_params = [
        p for n, p in model.named_parameters() if p.requires_grad and not n.startswith("head.")
    ]
    optimizer = torch.optim.AdamW(
        [{"params": adapter_params, "lr": 0.001}, {"params": model.head.parameters(), "lr": 0.01}],
        weight_decay=0.01,
    )
    rng = np.random.default_rng(seed)
    history, best_score, best_state, stale = [], -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for start in range(0, len(y_train), 32):
            # Fixed epoch permutation, not independent replacement samples.
            if start == 0:
                order = rng.permutation(len(y_train))
            idx = order[start : start + 32]
            inputs = {k: v[idx] for k, v in encoded.items()}
            loss = torch.nn.functional.cross_entropy(model(**inputs), target[idx])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1)
            optimizer.step()
            total_loss += loss.item() * len(idx)
        model.eval()
        selection = classification(infer(model, tok, texts["selection"]), y_select)
        row = {
            "epoch": epoch + 1,
            "loss": total_loss / len(y_train),
            "selection_macro_f1": selection["macro_f1"],
            "selection_accuracy": selection["accuracy"],
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if selection["macro_f1"] > best_score + 1e-6:
            best_score, best_state, stale = (
                selection["macro_f1"],
                copy.deepcopy(model.state_dict()),
                0,
            )
        else:
            stale += 1
        if stale >= 6:
            break
    model.load_state_dict(best_state)
    model.eval()
    if original_frozen != frozen_hash(model):
        raise AssertionError("Frozen pretrained parameters changed")
    model.save(root)
    # Independent classical baseline and a frozen pretrained encoder baseline.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=6000)
    train_matrix = vectorizer.fit_transform(texts["train"])
    lexical = LogisticRegression(C=5.0, max_iter=1500, random_state=seed).fit(train_matrix, y_train)
    lexical_spec = {
        "vocabulary": vectorizer.vocabulary_,
        "idf": vectorizer.idf_.tolist(),
        "coef": lexical.coef_.tolist(),
        "intercept": lexical.intercept_.tolist(),
    }
    # numpy integers in the vocabulary are converted explicitly for portable JSON.
    lexical_spec["vocabulary"] = {k: int(v) for k, v in lexical_spec["vocabulary"].items()}
    (root / "lexical.json").write_text(json.dumps(lexical_spec, separators=(",", ":")))
    pooled = {k: embeddings(root / "base", tok, v) for k, v in texts.items()}
    frozen = LogisticRegression(C=1.0, max_iter=1500, random_state=seed).fit(
        pooled["train"], y_train
    )
    frozen_spec = {"coef": frozen.coef_.tolist(), "intercept": frozen.intercept_.tolist()}
    (root / "frozen.json").write_text(json.dumps(frozen_spec, separators=(",", ":")))
    logits = {
        "lora": {k: infer(model, tok, v) for k, v in texts.items()},
        "lexical": {
            k: lexical.decision_function(vectorizer.transform(v)) for k, v in texts.items()
        },
        "frozen": {k: linear_logits(v, frozen_spec) for k, v in pooled.items()},
    }
    policies = {
        name: select_policy(
            values["calibration"],
            labels(parts["calibration"]),
            values["policy"],
            values["policy_oos"],
        )
        for name, values in logits.items()
    }
    (root / "policies.json").write_text(json.dumps(policies, indent=2))
    y_test = labels(parts["test"])
    results = {
        name: {
            **classification(values["test"], y_test),
            "routing": routing_metrics(values["test"], y_test, values["test_oos"], policies[name]),
        }
        for name, values in logits.items()
    }
    # Evaluation only: no test-derived parameter, temperature or threshold updates below.
    mistakes = []
    predicted = logits["lora"]["test"].argmax(axis=1)
    for i, row in enumerate(parts["test"]):
        if predicted[i] != y_test[i]:
            mistakes.append(
                {"text": row["text"], "expected": row["label"], "predicted": LABELS[predicted[i]]}
            )
    report = {
        "schema_version": 1,
        "data_source": "CLINC150 selected ten-intent task + original OOS evaluation",
        "dataset_sha256": data["dataset_sha256"],
        "source_sha256": data["source_sha256"],
        "source_commit": data["source_commit"],
        "seed": seed,
        "labels": LABELS,
        "base_model": BASE_ID,
        "base_revision": BASE_REVISION,
        "max_length": MAX_LENGTH,
        "split_sizes": {k: len(v) for k, v in parts.items()},
        "removed_overlaps": data["removed_overlaps"],
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_fraction": trainable / total,
        "frozen_parameters_unchanged": True,
        "frozen_weights_sha256": original_frozen,
        "selected_epoch": max(history, key=lambda r: r["selection_macro_f1"])["epoch"],
        "history": history,
        "policies": policies,
        "test": results,
        "lora_test_errors": mistakes,
        "training_seconds": time.perf_counter() - started,
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "threads": 2,
        },
    }
    report["model_version"] = fingerprint(
        {
            "adapter_sha256": hashlib.sha256(
                (root / "adapter/adapter_model.safetensors").read_bytes()
            ).hexdigest(),
            "head": json.loads((root / "head.json").read_text()),
            "policies": policies,
            "dataset_sha256": data["dataset_sha256"],
        }
    )[:16]
    (root / "report.json").write_text(json.dumps(report, indent=2))
    write_manifest(root)
    return report
