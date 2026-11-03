"""Actual LoRA on attention query/value projections, plus two measured baselines."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel, BertTokenizerFast

from .data import LABELS, MAX_LENGTH


def tokenizer(base):
    return BertTokenizerFast(vocab_file=str(Path(base) / "vocab.txt"), do_lower_case=True)


class AdapterRouter(torch.nn.Module):
    def __init__(self, base: Path, adapter: Path | None = None, rank=8):
        super().__init__()
        encoder = AutoModel.from_pretrained(
            base, local_files_only=True, use_safetensors=True, attn_implementation="eager"
        )
        if adapter is None:
            config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=rank,
                lora_alpha=2 * rank,
                lora_dropout=0.05,
                target_modules=["query", "value"],
                bias="none",
            )
            self.encoder = get_peft_model(encoder, config)
        else:
            self.encoder = PeftModel.from_pretrained(encoder, adapter, is_trainable=False)
        self.head = torch.nn.Linear(encoder.config.hidden_size, len(LABELS))

    def forward(self, **inputs):
        return self.head(self.encoder(**inputs).pooler_output)

    def save(self, directory):
        directory = Path(directory)
        self.encoder.save_pretrained(directory / "adapter", safe_serialization=True)
        config_path = directory / "adapter/adapter_config.json"
        cfg = json.loads(config_path.read_text())
        cfg["base_model_name_or_path"] = "google/bert_uncased_L-2_H-128_A-2"
        config_path.write_text(json.dumps(cfg, indent=2))
        (directory / "head.json").write_text(
            json.dumps({k: v.detach().tolist() for k, v in self.head.state_dict().items()})
        )

    @classmethod
    def load(cls, directory):
        root = Path(directory)
        model = cls(root / "base", root / "adapter")
        model.head.load_state_dict(
            {k: torch.tensor(v) for k, v in json.loads((root / "head.json").read_text()).items()}
        )
        return model.eval()


def tokenize(tok, texts):
    return tok(
        texts, padding="max_length", truncation=True, max_length=MAX_LENGTH, return_tensors="pt"
    )


def infer(model, tok, texts, batch_size=64):
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(texts), batch_size):
            encoded = tokenize(tok, texts[offset : offset + batch_size])
            rows.append(model(**encoded).numpy())
    return np.concatenate(rows)


def embeddings(base, tok, texts):
    model = AutoModel.from_pretrained(
        base, local_files_only=True, use_safetensors=True, attn_implementation="eager"
    ).eval()
    rows = []
    with torch.inference_mode():
        for offset in range(0, len(texts), 64):
            rows.append(model(**tokenize(tok, texts[offset : offset + 64])).pooler_output.numpy())
    return np.concatenate(rows)


def linear_logits(x, spec):
    return x @ np.array(spec["coef"]).T + np.array(spec["intercept"])


def tfidf_from_spec(spec):
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), sublinear_tf=True, vocabulary=spec["vocabulary"]
    )
    vectorizer.fit(["initialize fixed vocabulary"])
    vectorizer.idf_ = np.array(spec["idf"])
    return vectorizer


def probabilities(logits, temperature=1.0):
    z = np.asarray(logits, dtype=float) / temperature
    z -= z.max(axis=1, keepdims=True)
    value = np.exp(z)
    return value / value.sum(axis=1, keepdims=True)
