# Enterprise Training-Run Readiness (2026-08-18)

Product of three consecutive deep sweeps at HEAD `a6ec119`:
**Sweep 1** executed real training runs on a synthetic micro-dataset and
catalogued every failure empirically; **Sweep 2** audited the loop
internals (checkpointing, resume, early stopping, determinism, signals);
**Sweep 3** audited the MLOps surface (tracking, credentials, data
governance, CI, dependencies). Every claim below is either executed proof
or file:line evidence.

---

## A. FIXED during the sweeps (commit `a6ec119`, verified by execution)

| # | Defect | Proof |
|---|--------|-------|
| A1 | **Trainer could not run on paddle 3.3.x**: warpctc requires labels int32 + BOTH length tensors int64; trainer passed all-int32 → crash on first batch | Crash reproduced; fixed; trainer now completes runs |
| A2 | **log_softmax fed to warpctc**, which normalises internally (documented "unscaled probability sequence") | Contract verified in paddle source; raw logits now |
| A3 | **Early stopping unconditionally fatal**: improvement test compared `val > val + min_delta` (best updated first) → counter never reset → every run killed at exactly patience+1 epochs | Reproduced (runs with falling loss killed on schedule); pure `update_early_stopping()` helper, unit-tested |
| A4 | Second `is_best` always False → best-checkpoint branch of `save_checkpoint` was dead code | AST-pinned to one computation |
| A5 | `latest` checkpoint only written on conditional saves; **no `latest_info.json`** → resume-from-latest silently restarted epoch counter + LR warmup; **non-atomic writes** could truncate the resume file | `latest`+info now every epoch, atomic tmp+rename; resume verified: "Resumed from epoch 3" with scheduler intact |
| A6 | Stopping epoch was neither logged nor checkpointed (break before the save block) | Break now flag-driven after checkpointing |
| A7 | 2 bare excepts in the trainer | narrowed, AST-pinned |

**End-to-end verification:** trainer learns (loss 13.8 → 0.095; 11/12
exact memorization on the micro-set under repo-default hyperparameters);
two same-seed runs bit-identical; resume works; export to inference format
works. Suite: 380 passed, 11 skipped.

## B. CONSEQUENCE FOR HISTORY (recorded in LOGBOOK)

Every historical Optuna trial sampled patience 3-20 under defect A3 →
**no historical training run was ever allowed past ~21 epochs, regardless
of progress**. Combined with the decoder findings of `e3ca117`, the
41.86% (18/43) baseline is a floor obtained under a training guillotine
and a broken evaluation decoder. Re-running the study after these fixes
is expected to move the number; nothing is claimed until measured.

## C. OUTSTANDING — required for an enterprise training run

### C1. Data access (the external dependency you named)
- **DagsHub credentials**: `cp .env.example .env`, fill
  `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` (DVC S3 remote) and
  `DAGSHUB_USER_TOKEN`. `.dvc/config.local` is gitignored for the same
  purpose. Remote: `s3://Thundastormgod` via `dagshub.com` (configured in
  `.dvc/config`).
- **No DVC pointers exist in-repo** (`*.dvc` files: zero) → the 381-image
  dataset is not versioned here at all. Required: `dvc import`/`dvc add`
  of the dataset so every run's provenance can pin an exact data hash
  (the tracking layer already fingerprints `dataset_roots`).
- **Dataset split governance**: after pulling, splits MUST be produced by
  `utils/prepare_dataset.create_splits` (VIN-grouped; leakage raises) —
  never hand-rolled.

### C2. Trainer ↔ tracking integration (P1)
`vin-train finetune` never opens a tracked run — zero `start_run`/mlflow
references in `finetune_paddleocr.py`. The provenance layer exists and is
used by the Optuna tuner, but a direct enterprise training run currently
records commit/data/env provenance nowhere. Fix: wrap `train()` in
`tracking.start_run` (params from config, `dataset_roots` from the config
data dirs, per-epoch `log_metrics`, artifacts: config + final metrics +
best checkpoint path).

### C3. Early-stopping metric choice (P1)
Now-honest early stopping still watches **exact-match accuracy only**,
which sits at 0 for the entire blank-collapse phase (executed: loss falls
13.6 → 3.4 with exact-match pinned at 0). On a 43-image val set the metric
is also quantised to 1/43 steps. Fix: monitor val_loss (or char accuracy)
for patience, and/or add a `min_epochs` floor before early stopping may
fire.

### C4. Hyperparameter search space is part-poison (P1)
Executed: constant lr 3e-3 drives this architecture into the blank basin
and it never escapes (600 steps); lr 1e-3 learns. The Optuna space
samples `learning_rate` up to **0.01** (log-uniform 5e-4..1e-2) — a large
fraction of the space is unlearnable, and under the old A3 guillotine
those trials were also truncated. Narrow the LR upper bound (~2e-3) or
add a collapse detector (loss ≈ ln(34) for N epochs → prune trial).

### C5. Confidence semantics (P2)
An all-blank decode reports "Avg Confidence 92.59%" in the trainer's
final metrics (confidence of predicting nothing). Kept-timesteps-only
confidence (as in `onnx_inference` / `ctc_greedy_decode` kept positions)
should be used; empty decode → 0.0.

### C6. Dataset robustness (P2)
`VINRecognitionDataset.__getitem__` recurses to the next sample on an
unreadable image: silent duplicate sampling, and infinite recursion if
all images are bad. Fix: count and cap skips (fail the run above a
corruption threshold), log the skipped files into the run record.

### C7. Shutdown semantics (P3)
SIGTERM/SIGINT handling is graceful, but a mid-epoch interrupt loses the
current epoch (save happens at epoch end — acceptable, now that `latest`
is refreshed every epoch) and then export+final-metrics run on the
partial weights. Consider skipping export on interrupted runs or tagging
the metrics file `interrupted: true`.

### C8. Dependency truth (P2)
- `paddlepaddle>=3.0.0` is a **core** dependency in pyproject, but the
  dev venv was built without it and **CI installs a paddle-less test
  set** (`ci.yml:43-46`) — the 2 paddle-gated tests never run in CI, and
  the CTC dtype blocker (A1) is exactly the class of failure that setup
  hides. Add a CI job (or nightly) that installs paddle CPU and runs the
  paddle-marked tests + a 2-epoch micro-train smoke (the Sweep-1 probe is
  scriptable).
- `paddle2onnx` is required for ONNX export but only in the `[onnx]`
  extra; the runbook must state which extras a training box needs:
  `pip install -e ".[training,tracking,onnx]"` + core.

### C9. Config schema validation (P3)
The trainer validates architecture/loss consistency but raw `KeyError`s
on missing keys (`config['Global']['save_epoch_step']`). A schema check
with named errors at startup (all required keys, types, ranges — e.g.
lr bounds per C4) turns config mistakes into 1-second failures instead
of mid-run crashes.

### C10. Runbook (P3)
No TRAINING_RUNBOOK exists. Minimum content: environment setup (extras,
paddle install), credential setup (C1), data pull + split procedure,
launch command, what to monitor (val_loss AND exact-match; collapse
plateau at loss≈3.53), resume procedure (`--resume <dir>/latest` — now
correct), where results land (tracked run + `training_metrics.json`),
and the reproduce command (`vin-reproduce <run_id>`).

## D. Ordered plan for the first enterprise run

1. C1 credentials + `dvc` data pull + DVC pointers committed.
2. C2 tracking integration (every run provenance-tracked from day one).
3. C3 early-stop metric + C4 LR bound (protect the run budget).
4. CI paddle smoke job (C8) so A1-class breakage can't ship again.
5. Launch: `vin-train finetune --config configs/vin_finetune_config.yml`
   on the grouped splits; monitor; expect the blank-collapse phase;
   evaluate through the fixed `multi_model_evaluation` path only.
6. Re-run the Optuna study (fixed tuner + fixed trainer + fixed scorer)
   to establish the first trustworthy baseline; record in LOGBOOK.
