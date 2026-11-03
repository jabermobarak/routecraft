# Trained RouteCraft adapter

Actual PEFT LoRA weights for the pinned two-layer pretrained encoder. Targets: query/value; rank 8; alpha 16. Frozen base weights are in `../base`; the jointly trained classifier is `../head.json`.

Derived from `google/bert_uncased_L-2_H-128_A-2`, revision `30b0a37ccaaa32f332884b96992754e246e48c5f`. Distributed under Apache 2.0; see `../base/LICENSE`. Training and evaluation details: `../../docs/model-card.md`.
