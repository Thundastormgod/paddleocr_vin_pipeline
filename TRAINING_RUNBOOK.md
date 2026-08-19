# Training Runbook — VIN Recognition Fine-Tuning

Operational procedure for a provenance-tracked training run. Every step
here was executed on 2026-08-18 against the real dataset; nothing is
speculative.

## 1. Environment

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[training,tracking,onnx]"   # core deps include paddlepaddle
pytest -q                                     # suite must be green before training
```

Paddle needs CPU or CUDA; Apple MPS is not supported (falls back to CPU).

## 2. Credentials (one-time)

Tokens: https://dagshub.com/user/settings/tokens

```bash
cp .env.example .env         # fill DAGSHUB_USER_TOKEN (+ AWS_* for DVC)
```

The dagshub client can also cache a token via `dagshub login`
(`~/Library/Caches/dagshub/tokens` on macOS). NEVER commit `.env` or
`.dvc/config.local` — both are gitignored.

## 3. Data

Source: DagsHub Data Engine datasource `Thundastormgod/jlr-vin-ocr`.
Recognition set = the `data/ocr_*` trees (plate crops 2048x390 with
`*VIN*` text labels). Full scenes with YOLO boxes live in `data/train|val|test`
(detection - not used by this trainer).

Pull + stage (downloads ~1.2 GB, builds trainer label files, fixes any
upstream VIN leakage, and runs the canonical no-leakage gate):

- images land in `finetune_data/dagshub/data/ocr_*/images`
- label files written: `finetune_data/{train,val,test}_labels.txt`
  (format: `<path relative to finetune_data>\t<VIN>`)
- splits are VIN-grouped and verified disjoint
  (`utils/prepare_dataset.assert_no_vin_leakage` raises on any leak)

Dataset facts (2026-08-18 pull): 2,529 crops, 2,440 unique VINs, 100%
checksum-valid labels, zero unreadable images. Staged split:
2,387 train / 102 val / 40 test.

## 4. Launch

```bash
.venv/bin/python -m src.vin_ocr.training.finetune_paddleocr \
    --config configs/vin_finetune_config.yml
# or: vin-train finetune --config configs/vin_finetune_config.yml
```

The run opens a tracked MLflow run automatically (experiment
`vin_finetune`, SQLite store `mlflow.db`) recording commit, diff,
dependency versions and dataset fingerprints. The reproduce command is
printed at completion. If tracking is unavailable the run proceeds with a
loud warning.

Config is validated at startup with named errors (`validate_config`);
learning rates above 2e-3 are rejected — measured to collapse CTC
training into the blank basin permanently.

## 5. What to monitor

- `output/vin_rec_finetune/training_progress.json` — live status.
- **Blank-collapse phase is normal**: loss falls from ~13.9 and plateaus
  near ln(34)=3.53 while exact-match stays 0, then breaks out. On the
  full dataset (149 batches/epoch) breakout happened within ~3 epochs;
  val loss reached 0.91 by epoch 17.
- Early stopping is loss-aware: the patience counter resets when EITHER
  exact-match improves by `early_stopping_min_delta` OR val loss falls by
  `early_stopping_loss_min_delta`; nothing fires before
  `early_stopping_min_epochs`.
- A non-finite loss ABORTS the run immediately with a diagnostic
  (usually CTC geometry: timesteps < 17).

## 6. Checkpoints, interruption, resume

- `latest.pdparams/.pdopt/latest_info.json` refreshed EVERY epoch,
  written atomically.
- `epoch_N.*` every `save_epoch_step` epochs; `best_accuracy.pdparams`
  on validation exact-match improvement.
- Ctrl-C / SIGTERM stops gracefully at the next batch boundary; progress
  since the last epoch end is lost by design.
- Resume: `... --resume output/vin_rec_finetune/latest` — continues at
  the recorded epoch with optimizer + LR-scheduler state intact.

## 7. Evaluation

Evaluate ONLY through the fixed paths:

```bash
# held-out test set, alignment-based metrics, VIN-grouped split
.venv/bin/python -m src.vin_ocr.evaluation.evaluate \
    --data-dir finetune_data --labels finetune_data/test_labels.txt ...
# or multi-model comparison (errors reported as not_evaluated, never 0%)
.venv/bin/python -m src.vin_ocr.evaluation.multi_model_evaluation
```

Record the result in LOGBOOK.md with the MLflow run ID (template at the
bottom of the logbook). Numbers without a run ID do not exist.

## 8. Replay any historical run

```bash
vin-reproduce <run_id>          # or python -m src.vin_ocr.tracking.reproduce <run_id>
```

## 9. Model versions and traces (MLflow 3)

**Registry** — every promotable checkpoint becomes a version of the
registered model `vin-recognizer` (UI → Models tab). A version is a
pyfunc carrying checkpoint + config + charset + canonical decode - the
deployable unit, not a bare state dict. Metrics are MEASURED at
registration on a named dataset (never transcribed):

```python
from src.vin_ocr.tracking.model_registry import register_checkpoint_version
register_checkpoint_version(
    "output/vin_rec_finetune_stage3/best_val_loss.pdparams",
    stage_label="stage-3b-best-val-loss",
    source_run_id="<training run id>",
)
```

**Traces** — `vin-ocr recognize|batch` arms MLflow tracing automatically
when the [tracking] extra is installed (`--no-trace` to disable;
`MLFLOW_TRACKING_URI` honoured, else the repo SQLite store). Spans:
recognize (CHAIN) -> PaddleOCR.predict (TOOL) -> postprocess (PARSER).
Checkpoint-path tracing: `tracking.model_registry.traced_recognize()`
(preprocess/forward/decode/validate spans). View: UI -> vin_finetune ->
Traces tab.
