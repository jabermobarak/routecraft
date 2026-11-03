# Implementation notes

## Data → training → serving

`data.py` selects the public task and checks normalized-text disjointness. `train.py` fits three methods, selects the LoRA checkpoint using selection macro-F1, calibrates temperatures, chooses review thresholds, and finally evaluates test rows. It writes a report and a SHA256 manifest. `models.py` owns tokenization and safe serialization. `engine.py` verifies and loads the bundle with local-only model loading. `api.py` exposes batch inference; the static dashboard calls that API.

No Python pickle or joblib model is loaded. The base and adapter use safetensors. Baselines and the head use plain JSON. Bundle hashes detect accidental changes; they are integrity checks, not a signed supply-chain attestation. Replacing both files and manifest could bypass them.

A single process holds the LoRA model, a separate frozen base for the comparison, and the lexical baseline. A lock serializes model calls and aggregate counter updates. The reported request latency covers all three methods, tokenization and lock time, not a standalone LoRA speed benchmark. Multiple workers duplicate model memory and expose separate process counters. This demo does not implement a durable queue or distributed metrics.

## Design choices

**Why a small encoder?** It supports an actual CPU training run and a portable, inspectable repository. Its base weight file is about 17 MB. The project demonstrates parameter-efficient adaptation without requiring a hosted model, a GPU or a paid key.

**Why keep a lexical baseline?** The benchmark has short and often distinctive intent phrases. TF-IDF performs strongly. Showing its better in-scope accuracy avoids assuming a transformer must win.

**Why separate calibration and threshold splits?** Temperature fitting and review-threshold selection are different choices. Both need validation evidence, while the final test set stays untouched. The small splits make the estimates noisy; a larger task would benefit from repeated seeds and matched-coverage comparisons.

**What does LoRA change?** Low-rank matrices on attention query/value projections and the classification head. The pretrained encoder matrices remain frozen. Tests check an actual optimizer step and the full training report records a frozen-weight hash check.

## API operation

By default the server binds to localhost. Docker publishes its port only on host loopback. `RC_API_TOKEN` protects config/report/routing/metrics; the page supports entering the token in memory. Use TLS and an authenticated reverse proxy before exposing this beyond a trusted machine. There is no user-account system, rate limiting or abuse protection in this repository.

The readiness endpoint fails closed if the bundle is missing or its checksums are invalid. Input validation rejects extra fields, empty or oversized batches and strings outside the size limit. No external inference call or filesystem path can be supplied by a request.

## Interview walkthrough

Start with the task boundary and source split diagram in code. Show the frozen-weight test, adapter parameter count and epoch-selection history. Compare the three test rows and explain why lexical accuracy is higher. Demonstrate an in-scope order request and an out-of-scope request, then explain the coverage/rejection tradeoff. Finish with test-label mutation: the learned weights and policy must remain identical when only test labels change.

Possible next experiments: repeat seeds, include omitted intents as OOS negatives with separate evaluation, compare at equal in-scope coverage, and measure per-method latency and memory on fixed hardware. These are proposed extensions, not completed features.
