# Provenance status of these artifacts (2026-08-19)

These files are HISTORICAL and several are superseded. Read LOGBOOK.md
before citing any number here.

| Artifact | Status |
|---|---|
| `multi_model_evaluation.json` | **SUPERSEDED / UNTRUSTWORTHY** — produced before the evaluator fixes of 2026-08-18: ONNX rows were decoded with the wrong CTC blank index (the recorded 0.0% for the fine-tuned model measured the decoder, not the model), char metrics were positional and `'_'`-padded, and crashes were scored as empty predictions. |
| `model_comparison.csv`, `sample_results.csv` | Same generation as above — same defects. |
| `batch_evaluation_*.json` | Small-n (3-5 images) pipeline probes; honest as recorded but not baselines. |
| `experiment_summary.json` | The n=20 pilot. Precision/recall stored as the string `"improved"` — no numeric pipeline P/R was ever recorded. |
| `detailed_metrics.json` | Baseline P/R (41.7%/44.4%) computed; `detection_rate: 0.997` is HAND-ENTERED — no code path computes a detection rate. |
| `check_this_result.csv` | Scratch file. |

Current measured results live in MLflow (`sqlite:///mlflow.db`,
experiments `vin_finetune` / `checkpoint_validation`, registry
`vin-recognizer`) with full provenance; replay any run with
`vin-reproduce <run_id>`.
