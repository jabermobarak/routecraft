"""Offline-only bundle loading and bounded, serialized local inference."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel

from .data import LABELS, MAX_LENGTH, fingerprint
from .models import (
    AdapterRouter,
    linear_logits,
    probabilities,
    tfidf_from_spec,
    tokenize,
    tokenizer,
)


def verify_bundle(root):
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if fingerprint(manifest["files"])[:16] != manifest["bundle_version"]:
        raise ValueError("Manifest checksum mismatch")
    required = {
        "base/config.json",
        "base/vocab.txt",
        "base/model.safetensors",
        "adapter/adapter_config.json",
        "adapter/adapter_model.safetensors",
        "head.json",
        "frozen.json",
        "lexical.json",
        "policies.json",
        "report.json",
    }
    if not required <= set(manifest["files"]):
        raise ValueError("Incomplete model bundle")
    for name, expected in manifest["files"].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Manifest path leaves the bundle")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Bundle checksum mismatch: {name}")
    return manifest


class Engine:
    def __init__(self, root: Path):
        root = Path(root)
        self.manifest = verify_bundle(root)
        torch.set_num_threads(2)
        self.report = json.loads((root / "report.json").read_text())
        self.version = self.report["model_version"]
        self.policies = json.loads((root / "policies.json").read_text())
        self.tok = tokenizer(root / "base")
        self.adapted = AdapterRouter.load(root)
        self.frozen = AutoModel.from_pretrained(
            root / "base", local_files_only=True, use_safetensors=True, attn_implementation="eager"
        ).eval()
        self.frozen_spec = json.loads((root / "frozen.json").read_text())
        self.lexical_spec = json.loads((root / "lexical.json").read_text())
        self.lexical = tfidf_from_spec(self.lexical_spec)
        self.lock = threading.Lock()
        self.requests = 0
        self.texts_scored = 0
        self.total_seconds = 0.0

    def route(self, texts):
        if not 1 <= len(texts) <= 16:
            raise ValueError("A batch must contain 1..16 texts")
        if any(not isinstance(t, str) or not 2 <= len(t.strip()) <= 1000 for t in texts):
            raise ValueError("Texts must contain 2..1000 characters")
        texts = [t.strip() for t in texts]
        started = time.perf_counter()
        with self.lock, torch.inference_mode():
            encoded = tokenize(self.tok, texts)
            logits = {
                "lora": self.adapted(**encoded).numpy(),
                "frozen": linear_logits(
                    self.frozen(**encoded).pooler_output.numpy(), self.frozen_spec
                ),
                "lexical": linear_logits(self.lexical.transform(texts), self.lexical_spec),
            }
            models = {
                name: probabilities(values, self.policies[name]["temperature"])
                for name, values in logits.items()
            }
            token_counts = [
                len(self.tok.encode(t, add_special_tokens=True, truncation=False)) for t in texts
            ]
            rows = []
            for index, text in enumerate(texts):
                result = {
                    "text": text,
                    "tokens": token_counts[index],
                    "truncated": token_counts[index] > MAX_LENGTH,
                    "models": {},
                }
                for name, matrix in models.items():
                    p = matrix[index]
                    order = np.argsort(-p, kind="stable")[:3]
                    confidence = float(p[order[0]])
                    threshold = self.policies[name]["threshold"]
                    result["models"][name] = {
                        "predicted_intent": LABELS[order[0]],
                        "confidence": confidence,
                        "decision": "route" if confidence >= threshold else "review",
                        "threshold": threshold,
                        "temperature": self.policies[name]["temperature"],
                        "top_intents": [
                            {"intent": LABELS[i], "probability": float(p[i])} for i in order
                        ],
                    }
                rows.append(result)
            elapsed = time.perf_counter() - started
            self.requests += 1
            self.texts_scored += len(texts)
            self.total_seconds += elapsed
        return {
            "model_version": self.version,
            "latency_ms": round(elapsed * 1000, 3),
            "results": rows,
            "note": "Classification suggestions only; no external action is executed. Low confidence requests are flagged for review.",
        }
