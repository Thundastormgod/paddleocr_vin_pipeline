# VIN OCR Logbook

Running narrative of what was changed, why, and what it did to the numbers.

Every entry in this file is attached to MLflow runs created after it, as the
run description (see `src/vin_ocr/tracking/provenance.py`). Write the entry
*before* starting the run, so the intent is recorded alongside the result
rather than reconstructed afterwards.

## Rules for entries

1. **State the hypothesis before the run, not after.** "Cosine schedule should
   beat step decay because the LR floor was truncating late-epoch refinement"
   is a hypothesis. "Cosine was better" is a result. Record both, in that order.
2. **Quote sample sizes.** `41.86% (18/43)` is a measurement. `41.86%` is a
   number. At n=43 the 95% confidence interval on 18/43 is [0.28, 0.57]; any
   comparison that does not survive that interval is not a finding.
3. **Never record a number that was not measured.** If a figure comes from a
   document, a plan or an estimate, label it as such. This project has already
   shipped a headline 46.51% that no code ever computed.
4. **Link the run.** Paste the MLflow run ID. A claim without a run ID cannot
   be checked, and anything that cannot be checked is not evidence.

---

## Current measured baseline

Established from the 30 genuine Optuna trials in `optuna_results/`:

| Metric | Value | Sample |
|---|---|---|
| Exact match (best trial) | 41.86% | 18/43 |
| Character accuracy (best trial) | 90.15% | 43 images |

Corrections on record, for anyone reading older documents:

- **46.51% / 94.39% were never measured.** They are literals in
  `validate_architectures.py`, which imports only `yaml`, `json` and
  `pathlib` — it loads no model, dataset or checkpoint.
  `architecture_summary.json` now carries `measured: false`.
- **"5% (19/382)" and "25% (96/382)" were extrapolations.** The source record,
  `results/experiment_summary.json`, has `sample_size_for_metrics = 20`. The
  real measurements are 1/20 and 5/20. The `/382` denominators were never
  observed.
- **"Rosetta" is not implemented in this codebase.** It appears in
  documentation only.

---

## Entries

### 2026-08-18 — Scoring integrity: splits, corrector, metrics, decode, geometry

**Problem.** Six defects in the measurement path, each confirmed live by
direct execution before fixing (commits 574ab28, e3ca117, ee523b9):

1. `evaluation/evaluate.py:create_splits` shuffled image PATHS - the same
   VIN landed in train, val and test simultaneously, so the scoring path
   measured plates the model trained on (the leak already fixed in
   `utils/prepare_dataset.py`, left live in the scorer).
2. `RuleBasedCorrector` stripped leading X/Y/T as "artifacts":
   `correct_vin("YV1MS390X72123456")` returned 16 characters.
3. Character metrics existed in four drifted copies; identical input
   scored F1 0.67 / 0.95 / 0.4706 / 0.9412. Two copies were positional
   (one leading artifact scored a 94%-correct prediction at 0.059) and
   the multi-model copy structurally could not charge precision for
   missing/invalid/extra characters (precision 1.000 at recall 0.294).
4. Three CTC decoders used three blank indices (0, 1, 33). The blank=33
   copy sat in `run_onnx` - the LIVE path for every exported PaddleOCR
   model - and decoded canonically-encoded "1M8" as "020N090".
5. The multi-model dispatch registered `finetuned_deepseek_onnx` but
   dispatched on `deepseek_finetuned_onnx`; the fall-through scored
   ("", 0.0) per image. Crashes, unreadable images and failed model
   initialisation were likewise recorded as empty predictions inside the
   accuracy denominator.
6. SVTR_LCNet downsampled width 32x: T=10 timesteps for a 17-char CTC
   target - inf loss from batch 0, silently averaged into the epoch loss.

**Change.** Single implementations, all consumers delegating:
`core/char_metrics.py` (alignment-based TP/FP/FN, CER = editdist/len(ref),
char_accuracy = max(0, 1-CER)); `charset.ctc_greedy_decode` (blank 0,
collapse-then-strip); the split delegates to the VIN-grouped canonical
splitter; artifact stripping is one regex shared by identity. The
multi-model evaluator dispatches through one table, reports crashed images
under `evaluation_errors` (excluded from denominators) and unrunnable
models under `not_evaluated` - never as 0% rows. Non-finite losses now
raise at all four accumulation sites. 57 new regression tests.

**Effect on model metrics.** Definitions changed; numbers move.
- Historical `multi_model_evaluation.json` results are NOT comparable to
  new runs: the old character metrics were positional and '_'-padded, and
  every old ONNX row was decoded with the wrong blank index. In
  particular, **the recorded 0.0% for `output/vin_rec_finetune` measured
  the broken decoder, not the model** - that checkpoint has no valid
  evaluation on record and must be re-run.
- The 41.86% (18/43) Optuna baseline is unaffected (it comes from the
  training-side scorer's exact-match count, which did not change).
- Results produced through `--create-splits` before this date leaked VINs
  across splits and should be discarded.

### 2026-08-18 — Trial-score fabrication removed from both Optuna tuners

**Problem.** Both hyperparameter tuners could report accuracies that were
never measured.

- Root tuner (`optuna_tuning.py`): every trial was scored from
  `output/vin_rec_finetune/training_metrics.json` whenever that file merely
  existed — no exit-code check, no freshness check. Every trial overwrites
  that same fixed path, so a trial that crashed before writing was scored
  from the *previous* trial's file. Reproduced directly: after a successful
  trial recording 0.4186, a crashed trial returned 0.4186. Crashes were also
  returned as 0.0, which the TPE sampler cannot distinguish from a measured
  zero. The study was in-memory only: one interrupt discarded every
  completed trial.
- Package tuner (`src/vin_ocr/training/hyperparameter_tuning/`, the one the
  web UI and `vin-train tune` launch): the objectives passed an
  `epoch_callback` keyword that **no trainer accepts**, so every trial
  raised TypeError — and `except Exception: return 0.0` scored the crash as
  a measured zero. The DeepSeek path imported `DeepSeekFineTuner`, a class
  that has never existed (the trainer is `DeepSeekVINTrainer`), and
  mislabelled the resulting ImportError as a *pruned* trial. This tuner was
  structurally incapable of producing a genuine measurement. No
  `optimization_results.json`/`trial_history.csv` artifacts exist in the
  repo, so no recorded number originates from it.

**Change.** A trial that produced no measurement now raises
`TrialExecutionError` (one shared definition in
`src/vin_ocr/training/hyperparameter_tuning/errors.py`) and is recorded by
Optuna as FAILED — visible, excluded from the sampler, non-fatal to the
study. Root tuner: metrics file deleted before launch, non-zero exit fatal,
metrics mtime must post-date the launch; study persisted to SQLite; every
trial is a tracked MLflow run with full provenance. Package tuner: scores
`PaddleOCRScratchTrainer.train()`'s returned best accuracy; DeepSeek trials
read the `training_progress.json` the trainer's callback writes, with
existence/freshness/parse/numeric guards; a study with no measurements
reports no best (None, not 0.0). Pinned by 35 regression tests
(`tests/test_optuna_tuning.py`, `tests/test_hyperparameter_tuning_package.py`),
including an AST guard that no except handler in either tuner returns a
numeric literal.

**Effect on model metrics.** None yet — this changes what can be *recorded*,
not what is computed. The standing 41.86% (18/43) baseline came from the
root tuner's `optuna_results/` corpus; any trial in that corpus whose
training crashed may carry a neighbour's accuracy under its own
hyperparameters, so per-trial hyperparameter conclusions drawn from it are
suspect until re-measured under the fixed tuner.

### 2026-08-17 — Experiment tracking with mandatory provenance

**Problem.** Three consecutive audits found metrics that could not be traced to
the code that produced them: a hardcoded 46.51%, `# Simulated` training results
returned without calling `train()`, and sample sizes extrapolated by a factor
of 19. The common cause was structural, not careless — repo-wide, **zero** files
recorded a commit SHA alongside a metric, so no result could be checked against
the code that produced it. The 63 files in `optuna_results/` carry
hyperparameters and accuracies with no commit, no data hash and no timestamp.

**Change.** Added `src/vin_ocr/tracking/`. Every run opened with `start_run()`
records the commit, a diff that reconstructs the working tree from it, the
resolved versions of paddle/paddleocr/numpy/opencv, DVC hashes and dataset
fingerprints, and the command that replays it. `provenance_complete` and
`provenance_reproducible` are recorded on every run, so an unprovenanced run is
never indistinguishable from a provenanced one.

**Notes.**
- Untracked-file capture is read-only. The usual `git add -N .` mutates the
  caller's index mid-run; this diffs against `/dev/null` instead. Pinned by
  `TestCaptureIsReadOnly`.
- Backend is local SQLite, not the `./mlruns` file store: MLflow 3 put the
  filesystem tracking backend in maintenance mode and it now raises.
- Provenance capture is stdlib-only. Only the sink needs MLflow.

**Effect on model metrics.** None. This changes what is recorded, not what is
computed. No baseline number should move as a result of this entry.

### Template for the next entry

```
### YYYY-MM-DD — <one-line summary>

**Hypothesis.** <what you expect to change, and the mechanism>
**Change.** <what you actually changed>
**Runs.** <MLflow run IDs>
**Result.** <measurement with sample size, vs the baseline above>
**Verdict.** <kept / reverted / inconclusive at this n>
```
