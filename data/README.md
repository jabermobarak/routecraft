# Dataset attribution

`task.json` contains selected utterances from **CLINC150 Full**, introduced in:

Stefan Larson, Anish Mahendran, Joseph J. Peper, Christopher Clarke, Andrew Lee, Parker Hill, Jonathan K. Kummerfeld, Kevin Leach, Michael A. Laurenzano, Lingjia Tang, Jason Mars. 2019. *An Evaluation Dataset for Intent Classification and Out-of-Scope Prediction.* EMNLP-IJCNLP. https://aclanthology.org/D19-1131/

Source: https://github.com/clinc/oos-eval/tree/828f8093932c8fe6ca7936c3d2e52903b1c523de

Dataset card and license declaration: https://huggingface.co/datasets/clinc/clinc_oos

License: Creative Commons Attribution 3.0 Unported. Full terms in `SOURCE_LICENSE`.

Changes made: selected ten nonfinancial English intents, subdivided original validation records with seed 42, added split/provenance metadata and normalized-overlap checks. Utterance wording was preserved. Original train and test boundaries are retained. Original out-of-scope validation and test records are retained for policy selection and final evaluation respectively. These OOS examples do not cover all 140 omitted in-scope classes; the benchmark is not a full open-world evaluation.

Split sizes: 1,000 train, 100 checkpoint selection, 50 calibration, 50 in-scope policy, 100 OOS policy, 300 in-scope test, 1,000 OOS test. The initial source file SHA256 and selected-split fingerprint are recorded in `task.json` and `artifacts/report.json`. No dataset record is used in two roles after case/whitespace normalization.
