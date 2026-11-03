"""Preserve source splits, separate tuning roles, and remove exact text overlaps."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

LABELS = [
    "book_flight",
    "book_hotel",
    "cancel_reservation",
    "change_language",
    "order_status",
    "reset_settings",
    "restaurant_reservation",
    "schedule_meeting",
    "sync_device",
    "weather",
]
DATA_COMMIT = "828f8093932c8fe6ca7936c3d2e52903b1c523de"
BASE_REVISION = "30b0a37ccaaa32f332884b96992754e246e48c5f"
BASE_ID = "google/bert_uncased_L-2_H-128_A-2"
MAX_LENGTH = 48


def normalized(text):
    return " ".join(text.casefold().split())


def fingerprint(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_splits(splits):
    seen = set()
    for name, rows in splits.items():
        if not rows:
            raise ValueError(f"Empty split: {name}")
        for row in rows:
            if not isinstance(row["text"], str) or not row["text"].strip():
                raise ValueError("Empty or invalid text")
            if row["label"] not in LABELS + ["oos"]:
                raise ValueError("Unknown label")
            key = normalized(row["text"])
            if key in seen:
                raise ValueError("Normalized text overlaps across or within splits")
            seen.add(key)


def prepare(source: Path, output: Path, seed=42):
    raw = source.read_bytes()
    data = json.loads(raw)
    parts = {
        k: []
        for k in ("train", "selection", "calibration", "policy", "policy_oos", "test", "test_oos")
    }
    rng = random.Random(seed)
    for label in LABELS:
        parts["train"].extend({"text": t, "label": y} for t, y in data["train"] if y == label)
        validation = [{"text": t, "label": y} for t, y in data["val"] if y == label]
        rng.shuffle(validation)
        parts["selection"].extend(validation[:10])
        parts["calibration"].extend(validation[10:15])
        parts["policy"].extend(validation[15:])
        parts["test"].extend({"text": t, "label": y} for t, y in data["test"] if y == label)
    parts["policy_oos"] = [{"text": t, "label": "oos"} for t, _ in data["oos_val"]]
    parts["test_oos"] = [{"text": t, "label": "oos"} for t, _ in data["oos_test"]]
    seen, removed = set(), {}
    for name, rows in parts.items():
        unique = []
        for row in rows:
            key = normalized(row["text"])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        removed[name] = len(rows) - len(unique)
        parts[name] = unique
    validate_splits(parts)
    package = {
        "schema_version": 1,
        "labels": LABELS,
        "seed": seed,
        "source_commit": DATA_COMMIT,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "removed_overlaps": removed,
        "splits": parts,
    }
    package["dataset_sha256"] = fingerprint(parts)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(package, indent=2), encoding="utf-8")
    return package


def load(path):
    package = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_splits(package["splits"])
    if fingerprint(package["splits"]) != package["dataset_sha256"]:
        raise ValueError("Dataset checksum mismatch")
    return package
