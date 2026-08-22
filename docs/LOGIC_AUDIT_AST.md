# Logic Audit — Sweep 2: Deterministic AST Analysis

Date: 2026-08-21
Companion to `docs/LOGIC_AUDIT.md` (sweep-1 manual findings C1–C4, H1–H10, M1–M33, L1–L50,
and sweep-1b findings W-C1, W-H1…W-H4, W-M1…W-M13, W-L1…W-L12).

## Engines

| Engine | What it is | Coverage |
|--------|-----------|----------|
| `ast_logic_sweep.py` (custom, 20 rules AST01–AST20) | Pure-`ast` structural analyzer: identical branches, duplicate conditions, self-compare, constant conditions, unreachable code, exception-handler shadowing, duplicate handlers, mutable defaults, zip-truncation, silent swallows, self-assign, duplicate dict keys, assert-on-tuple, jump-in-finally, identical bool operands, `is`-literal, param-shadowing loops, literal-only compares, cross-corpus tuple-unpack arity checking, sys.path edits | 78 first-party files (root scripts + `src/` + `scripts/`), 0 parse errors; plus 29 test files |
| ruff 0.16.3 (curated logic rules) | F631/F632/F702/F811/F821/F823/F841, E711–E714/E722, B002–B035 (bugbear), SIM114/115/220–223/911, PLE\*, PLW0120/0127/0128/0129/0131/0711/1508/2101/3301, PLR0124/1704, RUF008/017/034 | same tree |
| MCP `pydantic-ai_audit_codebase` | independent AST walker (broad excepts, os.system, control-flow asserts) | `src/vin_ocr` (54 files), `scripts` (8 files) |

Totals: custom analyzer 79 findings (production) + 16 (tests); ruff 25; MCP 44.
Every BUG-LIKELY finding below was manually triaged against source.

---

## 1. BUG-LIKELY findings (custom analyzer) — triage

| Rule | Location | Finding | Triage |
|------|----------|---------|--------|
| AST01 identical-branches | `train_pipeline.py:332` | `elif c.upper() in self.VIN_CHARSET: out += c.upper() / else: out += c.upper()` — branches identical, charset test irrelevant | **REAL** — confirms sweep-1 L8 (invalid chars passed through instead of dropped) |
| AST06 handler-shadow | `src/vin_ocr/utils/hardware_utils.py:245` | `except ImportError` unreachable: handler #1 already catches `Exception` | **REAL** — confirms sweep-1 M27 |
| AST19 unpack-arity | `find_best_epoch.py:49` | unpacking 2 names from `validate()`; every definition of `validate` in the corpus returns a 3-tuple | **REAL** — confirms sweep-1 H8 (script structurally cannot work) |
| AST03 self-compare | `tests/test_tracking_provenance.py:394` | `fingerprint_directory(root).manifest_sha256 == fingerprint_directory(root).manifest_sha256` | **FALSE POSITIVE** — two independent calls; legitimate determinism test |

No findings for: AST02 duplicate-condition, AST04 constant-condition, AST05 unreachable-after-jump,
AST07 duplicate-except, AST08 mutable-default, AST11 self-assign, AST12 duplicate-dict-key,
AST13 assert-on-tuple, AST14 jump-in-finally, AST15 identical-bool-operands, AST16 is-literal,
AST18 literal-compare — the corpus is clean on those defect classes.

Known limitation: AST05 only detects code after a literal jump statement in the same block.
The two dead-code items found manually in sweep 1 (`finetune_paddleocr.py:1817` — return after
an if/else where both branches return; `vin_pipeline.py:1480` — return after `parser.error()`,
which raises SystemExit through a call) require path-sensitive CFG analysis and are documented
in the main audit (L27, L11).

## 2. REVIEW findings (custom analyzer) — triage

| Rule | Location(s) | Triage |
|------|-------------|--------|
| AST17 param-shadow-loop x3 | `analyze_vin_errors.py:119,193,199` — loop var `metrics` shadows the `metrics` parameter | **REAL (latent)** — confirms sweep-1 L40; currently harmless, one refactor from reading the wrong dict. Also flagged by ruff PLR1704. |
| AST10 silent-swallow x18 | `ocr_providers.py:707`; `finetune_deepseek.py:669`; `finetune_paddleocr.py:2826`; `gpu_utils.py:187,195`; `hardware_utils.py:173`; `web/app.py:266,275,1487,1565,1713,1722,1728,2074,2563,2565,2573`; `web/training_components.py:373` | **REVIEW** — each `except [Exception]: pass` hides real failures on live paths. The `hardware_utils.py:173` and `web/app.py` clusters overlap sweep-1 M27/W-findings; the remainder are mostly best-effort cleanup paths. Recommend: narrow the exception or log. |

## 3. INFO findings (custom analyzer) — inventory

**AST09 `zip()` without `strict=` — 30 sites.** This is the defect class behind sweep-1
C1/M22/M30/M33 (silent truncation/misalignment). Sites needing `strict=True` or explicit
length handling, by risk:

- Metric-bearing (HIGH value): `src/vin_ocr/evaluation/metrics.py:477,509,536,652,669`;
  `run_experiment.py:94`; `train_pipeline.py:296,310`; `analyze_vin_errors.py:60,220`;
  `src/vin_ocr/training/finetune_paddleocr.py:2719,2724`; `finetune_deepseek.py:544,550`;
  `train_from_scratch.py:1229,1691`; `finetune_torch.py:446`; `debug_validation.py:79`;
  `src/vin_ocr/core/vin_utils.py:705`.
- Result/label pairing: `src/vin_ocr/inference/onnx_inference.py:530` (pairs with sweep-1 H3
  misordering — truncation would *mask* the misalignment); `scripts/onnx_example.py:168`;
  `train_pipeline.py:178,204,243,424,436`; `web/app.py:701,1625`.
- Benign (equal-length by construction): `evaluate.py:544`; `analyze_vin_errors.py:282`.

**AST20 `sys.path` manipulation — 25 sites** (root scripts, `scripts/`, `evaluate.py:36`,
`validate_dataset.py:31`, `prepare_finetune_data.py:37`, `finetune_paddleocr.py:72,103,141`,
`web/app.py:42`, `multi_model_evaluation.py:44`, …). Three are actively wrong or hazardous and
already documented: `evaluate.py:36` (shadows top-level `metrics`/`errors`/`cli` — L32),
`validate_dataset.py:31` (adds a directory that cannot satisfy the imports that follow — L42),
`prepare_finetune_data.py:37` (same — L43). `train_vin_streaming.py:17` inserts the repo's
*grandparent* (C2). The rest work only because the process CWD is conventionally the repo root.

## 4. ruff logic-rule results — triage (25 findings)

| Code | Location | Triage |
|------|----------|--------|
| F821 undefined name `torch` | `ocr_providers.py:1006` | **LATENT** — `'torch.device'` string return annotation on `_select_device`; `torch` is only imported inside function bodies, so `typing.get_type_hints()` on this class would raise. Runtime-safe today (body uses the `torch_module` param). |
| B023 loop-var binding x2 | `src/vin_ocr/core/vin_decode.py:110,111` | **FALSE POSITIVE** — `add()` closes over `new_beams` but is redefined and fully consumed within the same loop iteration; late binding cannot bite. Confirms sweep-1 "clean" verdict for `vin_decode.py`. |
| PLR1704 arg redefinition x2 | `analyze_vin_errors.py:119,193` | REAL (latent) — same as AST17 above (L40). |
| F841 unused local x11 | `finetune_paddleocr.py:1833` (`targets = batch['text']`), `:2600` (`labels = batch['label']`); `train_from_scratch.py:1263` (`dummy_input`); `train_vin_streaming.py:214` (`trained_model`); `web/app.py:1018` (`state`), `:2584-2585` (`gpu_info`, `gpu_available`); `web/training_components.py:286` (`valid_devices`); `convert_all_to_onnx.py:182` (`batch`); `onnx_example.py:278`; `scripts/prepare_dataset.py:166` (`test_vins`); `finetune_deepseek.py:100`, `finetune_paddleocr.py:158` (unused `e`) | **DEAD STORES** — no live-path damage; `web/app.py:2584` means the System Health panel computes GPU info it never displays; `training_components.py:286` confirms sweep-1b W-L4 (dead device check). |
| E722 bare except x5 | `gpu_utils.py:157,187,195`; `hardware_utils.py:173,235` | REAL (latent) — overlaps AST10/MCP; documented L47/M27. |
| SIM114 | `validate_dataset.py:176` | Style-adjacent; the interesting defect at that site is the dead checksum counter (M26), not the mergeable branches. |

ruff on `tests/` + `conftest.py` with the same rule set: **all checks passed**.

## 5. MCP audit results — triage (44 findings)

- `os.system()` x6 (`tracking/environment.py:140`, `train_from_scratch.py:1626`,
  `hardware_utils.py:158,165,319`, `web/app.py:1996`) — command-injection surface is nil
  (fixed strings), but return values are unchecked → failures invisible. REVIEW.
- broad/bare except x24 — same population as AST10/E722 above.
- assert-as-control-flow x14 (`finetune_paddleocr.py:716,727,1307,1316`; `torch_rosetta.py:58,84,109,117`; `prepare_dataset.py:84`; `prepare_finetune_data.py:249`; `train_smoke_test.py` x9) — INFO: these
  guards vanish under `python -O`. The `prepare_dataset.py:84` one is `assert_no_vin_leakage`'s
  raise-site — that one is a real `raise ValueError`, not an assert (MCP flagged the wrapper
  call pattern); the `torch_rosetta`/`finetune_paddleocr` asserts encode CTC-feasibility
  invariants and would silently disappear under -O. Worth converting to explicit raises.

## 6. New defects surfaced by sweep 2 (not in sweep 1)

| ID | Severity | Location | Finding |
|----|----------|----------|---------|
| A1 | LOW (latent) | `ocr_providers.py:1006` | `'torch.device'` annotation references a name never imported at module level — any annotation introspection raises `NameError`. |
| A2 | LOW | `finetune_paddleocr.py:1833,2600` | Dead stores of batch text/labels in the train/eval loops — leftovers that obscure which tensor is the supervision source. |
| A3 | LOW | `web/app.py:2584-2585` | System Health computes `gpu_info`/`gpu_available` and never renders them. |
| A4 | INFO | 30 `zip()` sites, 24 broad-swallow sites, 6 unchecked `os.system` sites, 14 `-O`-stripped asserts | Systemic hygiene classes inventoried in §3–§5 with exact locations. |

## 7. Cross-validation: sweep-1 findings independently re-detected by machines

| Sweep-1 ID | Re-detected by |
|------------|----------------|
| H8 (`find_best_epoch` 3-tuple unpack) | AST19 |
| L8 (identical elif/else in `train_pipeline`) | AST01 |
| M27 (`hardware_utils` handler shadow) | AST06 |
| L40 (`metrics` param shadowing) | AST17 + PLR1704 |
| M22/M30/M33/C1-adjacent (zip truncation class) | AST09 inventory |
| L32/L42/L43/C2-adjacent (sys.path hazards) | AST20 inventory |
| L47/M27 (bare/broad swallows) | AST10 + E722 + MCP |
| W-L4 (dead `valid_devices` check) | F841 |

Machine sweeps found **no counter-evidence** against any sweep-1 finding, and the false
positives they produced (B023 in `vin_decode.py`, AST03 in the provenance test) both landed in
files sweep 1 had independently verified as clean — consistent triage in both directions.

## 8. Reproduction

```bash
# custom analyzer (stdlib only) - checked in at scripts/ast_logic_sweep.py
python scripts/ast_logic_sweep.py config.py train_pipeline.py train_vin_model.py \
  train_vin_streaming.py train_paddleocr.py optuna_tuning.py run_experiment.py \
  resume_training.py analyze_vin_errors.py find_best_epoch.py \
  find_incorrect_predictions.py debug_validation.py setup_dagshub_dvc.py \
  verify_resume_training.py conftest.py src scripts --json ast_findings.json

# ruff cross-check
ruff check --no-cache --isolated --select \
  F631,F632,F633,F702,F811,F821,F823,F841,E711,E712,E713,E714,E722,B002,B004,B005,B006,B008,B012,B014,B015,B016,B017,B018,B020,B021,B022,B023,B024,B025,B029,B031,B032,B033,B035,SIM114,SIM115,SIM220,SIM221,SIM222,SIM223,SIM911,PLE,PLW0120,PLW0127,PLW0128,PLW0129,PLW0131,PLW0711,PLW1508,PLW2101,PLW3301,PLR0124,PLR1704,RUF008,RUF017,RUF034 \
  config.py train_pipeline.py ... src scripts
```

The analyzer is checked in at `scripts/ast_logic_sweep.py` (rules AST01–AST20 as tabulated
in §Engines; stdlib-only — `ast`, `builtins`, `json`, `argparse`, `pathlib`; two passes:
corpus-wide return-arity collection, then per-file visitors; JSON + grouped-text output),
so this audit is reproducible from the repository alone.
