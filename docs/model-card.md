# RouteCraft model card

## Task and model

English intent classification into ten labels, with a confidence threshold for review. The base is `google/bert_uncased_L-2_H-128_A-2` at revision `30b0a37ccaaa32f332884b96992754e246e48c5f`: two layers, 128 hidden dimensions, two attention heads. LoRA targets query/value projections (rank 8, alpha 16, dropout 0.05). A 128→10 linear head is trained jointly. Maximum sequence length: 48 WordPiece tokens including special tokens.

The pretrained base is Apache 2.0. Source: https://huggingface.co/google/bert_uncased_L-2_H-128_A-2 . Original license included in `artifacts/base/LICENSE`. The adapter is derived from this model and is distributed under Apache 2.0. The base authors retain their copyright. The architecture comes from the BERT family; the base was not pretrained as part of this project.

## Training and validation

Train: 1,000 public examples, 100 per label. Seed 42. AdamW, batch 32, adapter learning rate 0.001, head learning rate 0.01, weight decay 0.01, gradient clipping 1.0. Up to 24 epochs, stopping after six epochs without selection macro-F1 improvement. Selected epoch 15 of 21 completed epochs. Frozen tensors are hashed before/after training. Exactly 9,482 parameters are trainable out of 4,395,402.

Original validation rows are split per label into 10 checkpoint-selection, 5 calibration and 5 threshold-policy records. The policy also sees 100 original OOS validation examples. Scalar temperature minimizes calibration NLL on a fixed grid. Threshold maximizes the mean of in-scope acceptance and OOS rejection on the policy split. Each baseline fits its own temperature and threshold. That objective balances ID/OOS detection, not the correctness of accepted intent predictions; production policy might require a different cost function.

The final 300 ID and 1,000 OOS test examples are excluded from all training and policy choices. A regression test retrains twice after changing only test labels and verifies byte-identical learned parameters and policies. Train/validation/test boundaries originate from the public source. The ten-intent task itself is a deliberate restricted scope, not a representative production distribution.

## Results

The complete machine-readable evaluation, confusion matrices and sample errors are in `artifacts/report.json`. The README gives the main table. LoRA accuracy 97.3% versus frozen 91.7% and lexical 98.7%. At its threshold of 0.91 and temperature 2.124, LoRA accepts 84.7% of in-scope test requests and 6.6% of OOS test requests. Its accepted-ID accuracy is 100% on 254 accepted ID records; that denominator **excludes** the 66 accepted OOS requests. It must not be interpreted as 100% overall routing accuracy.

## Limitations

One task, one seed, small calibration/policy sets and no confidence intervals. Confidence thresholds may shift with the input distribution. The public OOS set is not an exhaustive sample of unsupported requests. Omitted CLINC150 intents are not included as extra negative training data. No multilingual, adversarial, prompt-injection or production traffic benchmark was performed. Names, long requests, typos and ambiguous multi-intent requests may fail. Inputs over 48 tokens are truncated with an explicit flag.

This is neither a generative assistant nor a tool-executing agent. A `route` is a suggestion. No external action occurs. No personal data was collected for this project; the app does not persist request text. Deployment operators remain responsible for their own infrastructure logs and access controls.
