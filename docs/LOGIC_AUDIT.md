# Logic Audit — paddleocr_vin_pipeline

Date: 2026-08-21
Scope: all first-party Python under repo root, `src/`, `scripts/` (~30k LOC).
Method: Sweep 1 = five parallel manual/LLM review passes with empirical reproduction of
key defects, followed by direct source verification of every CRITICAL/HIGH finding.
Sweep 2 = deterministic AST analysis (custom analyzer + ruff logic rules + MCP AST audit),
results in the second half of this document.

Severity legend:
- **CRITICAL** — cannot run at all, or systematically corrupts data/results on a live path.
- **HIGH** — produces wrong results silently, or a primary workflow reliably fails.
- **MEDIUM** — wrong under realistic conditions; misleading outputs; dead safety logic.
- **LOW** — latent traps, dead code, misleading diagnostics; no current live-path damage.

Verification status: every CRITICAL and HIGH item below was verified line-by-line against
source (and several reproduced by execution). MEDIUM/LOW items were source-verified by
reviewers; a spot-checked subset was re-verified directly.

---

## 1. CRITICAL

### C1. `train_pipeline.py:291` — correction rules rewrite VALID VIN characters
`_build_correction_rules` seeds `{"S":"5","B":"8","G":"6","Z":"2", ...}`. S, B, G, Z are
legal VIN characters (only I/O/Q are excluded). `_apply_rules` (line 330) checks
`if c in rules` first, so every correctly-recognized S/B/G/Z in every prediction is
rewritten. Every `SAL…` (Land Rover) VIN in this dataset is corrupted to `5AL…` by the
"correction" step. Learned rules (296–305) are mined via positional `zip(gt, pred)`, so a
single insertion/deletion misaligns all subsequent positions and generates garbage rules.
Fix direction: seed rules must contain only I/O/Q→digit maps; learned rules require
alignment-based mining (edit-distance opcodes), purity thresholds vs true-positive counts,
and checksum-gated application.

### C2. `train_vin_streaming.py:19` — import of nonexistent module; script dead on arrival
`from src.vin_ocr.data.setup_dagshub_streaming import ...` — `src/vin_ocr/` has no `data/`
package. `ModuleNotFoundError` at launch. Even if it imported: line 109 passes a config
*path string* to `VINFineTuner(config: Dict)` → `TypeError` at init (`config['Global']`);
line 74–89 `config.update(streaming_paths)` would wipe `Train.loader`; line 92
`.replace('.yml', …)` no-ops for `.yaml` and overwrites the original config; line 16
`project_root = Path(__file__).parent.parent.parent` points two directories above the
repo; `--cpu` (160–164) is parsed and ignored.

### C3. `run_experiment.py:276` — import of nonexistent root `prepare_dataset`
`from prepare_dataset import prepare_dataset` — there is no root `prepare_dataset.py`
(implementations live at `src/vin_ocr/utils/prepare_dataset.py` and
`scripts/prepare_dataset.py`; only the repo root is on `sys.path`, line 179).
`run_full_experiment()` raises `ModuleNotFoundError` at Step 1, every time.

### C4. `src/vin_ocr/training/train_from_scratch.py` — LR scheduler never stepped (paddle trainer)
`self.scheduler` is constructed (line 836) wrapped in
`LinearWarmup(start_lr=learning_rate*0.01)` (828–834), but the paddle training loop
(~910–995) contains no `scheduler.step()` — the only `.step()` (line 1507) belongs to the
separate torch DeepSeek trainer. Paddle schedulers advance only via explicit `.step()`,
so with the default config (warmup_epochs=5) the entire run trains at `start_lr` = **1% of
the configured LR**; with warmup disabled, decay schedules never decay. All
`lr_scheduler`/`warmup_epochs` config values are inert.
Related: scheduler horizon is built from a guessed `max(100, 1000 // batch_size)` steps
per epoch (805–810) instead of `len(train_loader)` — wrong by ~8x on the advertised
dataset; must be fixed together with the missing `.step()`.

---

## 2. HIGH

### H1. `src/vin_ocr/providers/ocr_providers.py:1142,1438-1440,1756` — fabricated confidences win ensembles
`_calculate_confidence(vin_text, text)` is never passed logits (all 3 call sites), so the
model-confidence branch (1428–1434) is dead and every format-valid DeepSeek read gets a
flat 0.85 (0.95 if checksum-valid). `_best_strategy` = `max(results, key=r.confidence)`
compares these constants against PaddleOCR's genuine softmax scores: a wrong-but-well-
formed read at 0.85 beats a correct real read at 0.84. Same inflated constants feed
`weighted_vote`/`weighted_char_vote`.

### H2. `ocr_providers.py:203-204,759,1099,1251` — `backend="vllm"` crashes at recognize
`__post_init__` maps `use_vllm→backend` one-directionally. `initialize()` gates on
`backend == "vllm" or use_vllm` and loads vLLM; `recognize()`/`recognize_batch()` gate on
`use_vllm and _vllm_model` → route to `_recognize_transformers` where `_model is None` →
`AttributeError` on every call. This is the exact usage shown in the class docstring.

### H3. `src/vin_ocr/inference/onnx_inference.py:371-405,530` — batch results misordered on mid-batch failure
Preprocess-failure rows are appended to `results` during the preprocess loop; successes
are appended after inference. For `[A(ok), B(err), C(ok)]`, `results` =
`[err_B, ok_A, ok_C]` — no longer index-aligned with `images`. The CLI `zip(images,
results)` (530) then reports B's error under A's filename and A's VIN under B's. Silent
misattribution of VINs to images.

### H4. `src/vin_ocr/training/finetune_paddleocr.py:1728,1755-1761` — validation dataset built with the Train data_dir
`data_dir = train_config['dataset']['data_dir']` is used to build BOTH datasets;
`Eval.dataset.data_dir` is required by `_REQUIRED_CONFIG_KEYS` (207) then ignored. Missing
val paths are silently dropped by `_load_samples` (381), so if roots differ, checkpoint
selection and early stopping run on a shrunken or empty val set.

### H5. `src/vin_ocr/core/vin_utils.py` — the public `correct_vin()` corrector is internally broken
- **626–665**: `_extract_vin_candidate` reintroduces the exact extraction bug the module's
  own docstring (362–373) documents as fixed: first-WMI-match wins, the `>10` score guard
  is a no-op (17 valid chars score ≥34), and the stale `_score_candidate` copy lacks the
  +100 checksum bonus present in `extract_vin_from_text`. Reproduced: `correct()` returns
  a checksum-invalid window while `extract_vin_from_text` on the same input returns the
  checksum-valid VIN.
- **675–678, 581**: `_apply_position_rules` rewrites letters at positions 12–14 to digits
  unconditionally (49 CFR 565 allows alphanumerics there for small-volume manufacturers),
  breaking already-checksum-valid VINs; validation happens after, with no rollback.
  Reproduced: valid `5FNRL6H09LBB00001` → invalid `5FNRL6H09LB800001`.
- **705–721**: `learn_from_errors` mines rules by positional `zip` (misalignment,
  truncation), has no purity check vs true-positive counts (2 errors suffice), applies
  rules globally including the check digit, and poisons the module-global singleton
  (`get_corrector()`, 759) process-wide. Reproduced: two misaligned samples caused a
  perfect read to be corrupted to garbage.

### H6. `resume_training.py` — both resume paths broken
- **48/54**: passes `best_accuracy.pdparams` WITH extension; `load_checkpoint`
  (finetune_paddleocr.py:2213) derives `best_accuracy.pdparams_info.json`, never written
  (real file: `best_accuracy_info.json`) → weights load, but epoch/global_step/best
  silently reset to 0; script prints "RESUME ENFORCED" while the resume state is
  discarded. Also `_save_best_model` writes no `.pdopt`, so optimizer state is always
  fresh on this path.
- **129–141**: when `--checkpoint` doesn't exist, the earlier fallback already appended
  the BEST checkpoint to `cmd`; the second check then runs `cmd.remove(args.checkpoint)` →
  `ValueError` (the bad path was never in `cmd`). If no fallback existed,
  `cmd.remove("--resume")` raises instead.

### H7. `train_pipeline.py:492-493` — `--no-gpu` finetune passes `--cpu`, which the trainer's argparse rejects
`finetune_paddleocr.py`'s parser (3085–3113) defines only `--config/--resume/--export-onnx`
(+ optional dagshub flags). argparse exits code 2; `--method finetune --no-gpu` always
fails after data prep. (GPU intent is already carried via `config['Global']['use_gpu']`.)

### H8. `find_best_epoch.py:49` — 2-variable unpack of a 3-tuple; script always returns None
`val_loss, val_acc = trainer.validate()` but `validate() -> Tuple[float, float, float]`
(finetune_paddleocr.py:1935). The per-epoch `except Exception` (76) swallows the
`ValueError`, printing "Epoch N: ERROR"; no epoch ever scores; `best_epoch=None`,
`best_correct=0` — total failure dressed as a result.

### H9. `src/vin_ocr/utils/prepare_finetune_data.py:279-285` — per-image split leaks same-VIN images across train/val
`random.shuffle(samples)` then index split operates on image rows, not VIN groups.
Multiple images of one VIN straddle both splits — the exact leakage
`assert_no_vin_leakage` (src/vin_ocr/utils/prepare_dataset.py:73-83) exists to prevent.
Validation accuracy from this entry point is inflated by memorization.

### H10. `src/vin_ocr/evaluation/multi_model_evaluation.py:1005-1013 vs 1296-1300, 1389-1399` — error rows crash both consumers
Error rows are appended with a reduced schema (no `prediction`, `exact_match`,
`chars_correct`, `char_accuracy`, `match_pattern`, `confidence`, `processing_time`), but
`_print_comparison` and the CSV writer index those keys directly → one unreadable image
(the exact case the error machinery exists for) raises KeyError mid-report; results are
lost / CSV truncated.

---

## 3. MEDIUM

| ID | Location | Problem |
|----|----------|---------|
| M1 | `src/vin_ocr/pipeline/vin_pipeline.py:558-565, 719, 787, 1300, 1368` | `_timer` writes `elapsed['ms']` after `yield`, but results copy the float inside the with-block → `processing_time_ms` is always 0.0 in both pipelines (reproduced). |
| M2 | `vin_pipeline.py:1337,1343` + `ocr_providers.py:108,540-545,1590-1596` | Multi-provider path preprocesses every image TWICE (full CLAHE/morph/bilateral chain in pipeline, then again inside provider). Factory `_create_config` silently drops `preprocess_enabled`, so callers cannot disable it. |
| M3 | `ocr_providers.py:1762-1768` | `_vote_strategy` never votes: `Counter.most_common(1)` ties break by insertion order → 2-provider disagreements always return provider[0]. |
| M4 | `src/vin_ocr/inference/paddle_inference.py:264-265` | Deletes I/O/Q instead of mapping (I→1, O→0, Q→0 policy everywhere else), left-shifting chars, then `[:17]` truncates → plausible-but-wrong VIN, no error. |
| M5 | `ocr_providers.py:1363-1366` | DeepSeek `_extract_vin` takes the FIRST 17-char regex window; a leading artifact char shifts the VIN and drops its last char; downstream window extraction no-ops at len==17. Bypasses the checksum-scored `extract_vin_from_text`. |
| M6 | `src/vin_ocr/preprocessing/vin_preprocessor.py:357-373` | Resize: width forced to `target_width` while height clamps to [32,512] → aspect ratio silently broken on extreme inputs despite `maintain_aspect=True`; the `maintain_aspect=False` branch resizes to `(target_width, min_height)` — min used as THE height. |
| M7 | `train_from_scratch.py:1479-1494` | Grad accumulation: loss never divided by accumulation steps (effective step 8x); leftover partial-window grads leak into the next epoch. |
| M8 | `train_from_scratch.py:1405 vs 1604` | Generation seeded with `<PAD>` (token 0); training teacher-forcing starts with `<SOS>` (token 1). Train/eval token asymmetry. |
| M9 | `train_from_scratch.py:604-626` | SVTR branch: `flatten(2)` produces an h-major sequence; `AdaptiveAvgPool1D(80)` bins mix rows → CTC timesteps do not correspond to image columns (violates monotonic alignment premise). |
| M10 | `train_from_scratch.py:961-977` | Validation/best-save only every 500 global steps, never at epoch end → small runs never validate, never write best_model. |
| M11 | `train_from_scratch.py:194` | `seed` config field dead: nothing seeds paddle/numpy/random in this module. |
| M12 | `train_from_scratch.py:1083,1593` | Corrupt-image handling by unbounded recursion `__getitem__(idx+1)` → RecursionError on runs of bad files; silent neighbor duplication otherwise. (finetune_paddleocr.py:469-501 has the fixed bounded form.) |
| M13 | `finetune_paddleocr.py:385-392,514` | Unknown label chars encode as 0 = CTC blank INSIDE targets (invalid for warpctc/torch CTC); `label_length` still counts them. In CE path, 0 = ignore_index → silently unsupervised positions. |
| M14 | `finetune_paddleocr.py:1644-1646` | `Const`/`Constant` scheduler branch instantiates Paddle's abstract `LRScheduler` base → `NotImplementedError` at optimizer construction. Always-broken branch. |
| M15 | `finetune_paddleocr.py:2141-2148,2213-2219` | Resume never saves/restores `best_val_loss`/`best_val_char_accuracy`/patience → first post-resume epoch overwrites best checkpoints; early-stop baselines reset. |
| M16 | `finetune_paddleocr.py:2213` | Resume with explicit `.pdparams` path derives `<name>.pdparams_info.json` (never written) → epoch counter silently lost (same root cause as H6a). |
| M17 | `finetune_paddleocr.py:2094-2101,2208-2211` | `best_accuracy` checkpoint has no `.pdopt`; resume silently proceeds without optimizer/scheduler state (half-resumed run). |
| M18 | `finetune_paddleocr.py:2467-2470,2943-2947` | `export_inference_model()` mutates `self.model` by loading `best_accuracy.pdparams` (possibly STALE from a previous run in the same dir) before `_save_final_metrics()` → final metrics can describe another run's weights. |
| M19 | `finetune_torch.py:226-235 vs 445-448,596-598` | Training masks pad timesteps via per-sample CTC input lengths, but eval argmax-decodes all 80 timesteps (violates the repo's own decode contract for pad regions). |
| M20 | `src/vin_ocr/evaluation/metrics.py:578-597` | `normalized_edit_distance` ≡ `char_error_rate` (same numerator & denominator, two names) with contradictory empty-set fallbacks (0.0 vs 1.0); also incompatible with evaluate.py's per-sample NED (542–546) — same field name, different math. |
| M21 | `evaluation/metrics.py:652-674 vs 541-545` | Per-class accuracy and confusion pairs computed POSITIONALLY, contradicting the module's own design comment; leading-artifact insertions fabricate the "most confused pairs" table. |
| M22 | `evaluation/metrics.py:477` | `add_batch` zips predictions/ground-truth without `strict=True` → silent truncation on mismatched lists (evaluate.py uses strict everywhere). |
| M23 | `multi_model_evaluation.py:548-576,1178-1189` | Default eval set pools `dagshub_data/train/images` with test images while scoring fine-tuned models → fine-tuned models partly evaluated on their own training data. |
| M24 | `multi_model_evaluation.py:815-852` | DeepSeek-ONNX branch: (a) casts raw logit floats to int as CTC class indices (no argmax) when C≤100; (b) "decodes" token IDs as ASCII, dropping IDs ≥127; (c) `str(output)` fallback lets `extract_vin_from_text` fabricate predictions from array reprs; (d) confidence = mean raw logit, printed as probability (e.g. 830%). |
| M25 | `optuna_tuning.py:372-374` | `study.best_trial is not None` cannot guard: Optuna RAISES ValueError when no trial has completed; callbacks are not covered by `catch=` → a failed trial 0 (or 5, 10 with zero completions) aborts the whole study, defeating the documented crash-isolation design. |
| M26 | `src/vin_ocr/utils/validate_dataset.py:186-189,306-317` | `report.invalid_checksum` is provably always 0: checksum failure never sets `is_valid=False`, and the increment lives in the branch reachable only for length/char failures. Dead safety counter. |
| M27 | `src/vin_ocr/utils/hardware_utils.py:243-246` | `except Exception` before `except ImportError` → second handler unreachable; the broad handler also silently flips `torch_available=False` on real (non-import) bugs the inner block was rewritten to propagate. |
| M28 | `hardware_utils.py:253,316` | `paddle_gpu = paddle.is_compiled_with_cuda()` (compiled ≠ present, no `device_count()`), consumed as `use_gpu` → selects `gpu` device on GPU-less machines; trainer init fails. |
| M29 | `scripts/prepare_dataset.py:114-129,282-287` | Existing labels keyed `train/x.jpg` can never match bare-filename lookups (`img.name`); label files loaded as dead weight; prepared-format images silently dropped. `get_all_images` is non-recursive → images in subdirs never found. |
| M30 | `run_experiment.py:94-117` | CER/substitutions/precision/recall/F1 computed by positional zip, not edit distance: one deletion counts as ~16 substitutions + 1 deletion (CER ~100% vs true ~6%). `levenshtein_distance` is imported and used for NED four lines later. |
| M31 | `debug_validation.py:62-65, 99-103` | `_preprocess_image` returns `(tensor, valid_width)` tuple; `image[np.newaxis, ...]` on a tuple → TypeError (script can never run). Separately: "Results match" is printed for ANY nonzero accuracy (20% vs 44% passes). |
| M32 | `analyze_vin_errors.py:44-69,139` | `insertion`/`deletion` counters structurally unreachable (only ed==1→substitution, ed>1→multiple); a single-char deletion is labeled "substitution". Executive summary hardcodes `/43` denominator. |
| M33 | `train_pipeline.py:296` | Confusion mining `zip(r['gt'], r['pred'])` truncates/misaligns on length mismatch (feeds C1's learned rules). |

## 4. LOW (latent traps, dead logic, misleading diagnostics)

| ID | Location | Problem |
|----|----------|---------|
| L1 | `vin_utils.py:148,156,162-164` | `_is_valid_vin_chars` returns true for I/O/Q (`VALID ∪ INVALID` = everything); labeled filename patterns can emit impossible ground truths; fallback-branch I/O/Q guard is dead. |
| L2 | `vin_utils.py:498-511` | Lowercase keys in `SEQUENTIAL_POSITION_RULES` unreachable (input uppercased at step 1). |
| L3 | `vin_utils.py:263` | `invalid_chars` lists only I/O/Q; `*`/`-` flag `has_valid_chars=False` yet appear nowhere in the diagnostic. |
| L4 | `vin_utils.py:520,744,751` | `learned_rules` aliasing: caller's dict mutated; `export_rules()` returns the live dict; mutating export desyncs `_global_rules`; empty-vs-nonempty dict semantics differ (`or {}`). |
| L5 | `src/vin_ocr/core/char_metrics.py:233-237` | `pairs` iterated twice; a generator input yields cer=0.0/acc=1.0 alongside tp=0/fp/fn>0 — silent self-contradictory metrics (reproduced). |
| L6 | `src/vin_ocr/core/charset.py:103-107` | Explicit-but-missing `dict_path` silently falls back to built-in charset (docstring scopes fallback to `None` only) — the silent char↔index mismatch class this module exists to prevent. |
| L7 | `charset.py:123-127` | Duplicate dict entries silently desync `num_classes` vs `idx_to_char` (last-wins mapping; reproduced). |
| L8 | `vin_pipeline.py:736,742,1318,1321` | `ImageLoadError(f"Image file not found: {path}")` passes the message as the `file_path` param → garbage context and "Reason: Unknown error". |
| L9 | `vin_pipeline.py:777-784,1358-1365` | `enable_postprocess=False` hardcodes `checksum_valid: False` — "not checked" conflated with "invalid". |
| L10 | `vin_pipeline.py:418-428` | `_fix_invalid_chars` runs on the whole raw string BEFORE window extraction: label noise ("MOTOR"→"M0T0R") becomes VIN-valid and biases window scoring when the true check digit was misread. |
| L11 | `vin_pipeline.py:1480-1481` | `parser.error()` raises SystemExit → `return 1` unreachable. `ARTIFACT_CHARS` imported unused; `VINResult` class dead. |
| L12 | `vin_preprocessor.py:189-410` | Every strategy assumes 3-channel input (`COLOR_BGR2GRAY`); 2-D grayscale arrays reach it via `OCRProvider._preprocess_image` (no channel guard) → cv2.error outside the provider's try. |
| L13 | `vin_preprocessor.py:454-457` | `image = cv2.imread(str(image))` shadows the path; failure message prints `None` instead of the path. |
| L14 | `onnx_inference.py:47` | `VIN_CHARSET_WITH_BLANK = VIN_CHARSET + " "` puts blank at index 33; repo convention (enforced by `load_char_dict`) is blank=0. Dead constant, loaded trap. |
| L15 | `onnx_inference.py:327` | `vin = raw_text[:17]` first-17 truncation, no scoring: a leading artifact yields a shifted "valid" VIN. |
| L16 | `ocr_providers.py:122,494-501` | `PaddleOCRConfig.use_gpu` stored, never acted on — GPU requests via provider path silently ignored. |
| L17 | `ocr_providers.py:1408-1411` | Wrong-length confidence `min(len,17)/17`: any TOO-LONG text scores band max 0.30, outranking a 16-char near-miss (0.288). |
| L18 | `ocr_providers.py:1084-1093` | DeepSeek preprocessing only for ndarray inputs; str/Path bypass it — `preprocess_enabled` silently inert for paths. |
| L19 | `ocr_providers.py:185 vs 1609` | `max_tokens` default 8192 (dataclass) vs 128 (factory) — truncation can cut output before the VIN appears. |
| L20 | `ocr_providers.py:1582-1614` | `_create_config` silently drops unknown kwargs (timeout, retries, preprocess flags, typos) — also what makes M2 unfixable from callers. |
| L21 | `ocr_providers.py:780-892` | Dead ONNX methods reference config fields that don't exist (`onnx_model_path` etc.) → guaranteed AttributeError if ever wired; `_export_to_onnx` both falls back AND raises. |
| L22 | `train_from_scratch.py:980,1100-1104,128,132,102` | `epoch_loss/num_batches` unguarded (ZeroDivision with drop_last on tiny sets); unknown label chars silently dropped (shortened CTC targets); `aug_prob`/`label_smoothing`/`backbone` config knobs inert. |
| L23 | `finetune_paddleocr.py:128,1900-1904` | `use_amp` never autocasts forward (auto_cast imported, unused) — AMP inert, scale/unscale overhead only. |
| L24 | `finetune_paddleocr.py:1684-1707` | AdamW decay exclusion via per-param `regularizer` is ineffective (decoupled decay honors only `apply_decay_param_fun`) — norm/bias still decayed while logged as excluded. |
| L25 | `finetune_paddleocr.py:777-779` | CTC input-length clamp guarantees `input>=label` but not `input>=label+adjacent-repeats` → rare infeasible alignment → inf loss kills run. |
| L26 | `finetune_paddleocr.py:2010`; `finetune_torch.py:429,451` | Epoch val loss = mean of unequal batch means (drop_last=False) — small final batch overweighted. |
| L27 | `finetune_paddleocr.py:1817` | Unreachable `return decoded` (both branches return earlier). |
| L28 | `finetune_torch.py:348-392` | Early-stop patience not persisted/restored across resume. |
| L29 | `finetune_torch.py:683-686` | MLflow dataset entity parses labels tab-only while loaders accept tab-or-space → garbage `vin` column in tracking metadata. |
| L30 | `evaluation/metrics.py:546-556` | Per-position denominators extend past the reference; fabricates positions >17 from insertions. |
| L31 | `evaluate.py:267-278,479` | Tab-less split line → `ground_truths[path]=None` → `len(None)` TypeError OUTSIDE the try, aborting the run. |
| L32 | `evaluate.py:36,47,38` | Useless `sys.path.insert(evaluation/)` shadows top-level `metrics`/`errors`/`cli`; unused `config`/`validate_vin` imports defeat the documented lazy-import policy and can break the console entry point from other CWDs. |
| L33 | `tracking/model_registry.py:366-368` | Registered version determined by post-hoc `max(search_model_versions)` — race attaches tags to the wrong version; ValueError on empty. |
| L34 | `tracking/dataset.py:126-148` | `_slug` collisions (`data/train_images` vs `data_train_images`) silently overwrite MLflow provenance params. |
| L35 | `optuna_tuning.py:54,465` | `TRIAL_METRICS_PATH` hardcoded independent of `--base-config`'s `save_model_dir` → mismatched config silently scores all trials as "wrote no metrics". |
| L36 | `optuna_tuning.py` | All trials share one `save_model_dir` → each trial overwrites the previous trial's checkpoints; best trial's weights unrecoverable. |
| L37 | `resume_training.py:43,124,146-151` | Dead `os.chdir` contradicted by `cwd=`; bare `"python"` instead of `sys.executable`; shadowed `config_file`. |
| L38 | `find_best_epoch.py:82-94` | Hardcoded `/43` denominators and `>=15`/`>=22` thresholds baked to one historical val set. |
| L39 | `debug_validation.py:43` | Reports "GPU" from `is_compiled_with_cuda()` (compile flag, not runtime device). |
| L40 | `analyze_vin_errors.py:60,237-339,119-199` | Positional zip inflates `error_positions`; hardcoded narrative ("Position 6: 65.1%", "A→F most common") presented as computed; `metrics` shadowed by loop vars. |
| L41 | `find_incorrect_predictions.py:40,53-75` | ZeroDivision on empty results; hardcoded `/17` and "samples 11-43" narrative. |
| L42 | `validate_dataset.py:115,31,307` | Plain-text label detection keyed on `'SAL'` substring; useless sys.path insert; unique-VIN counts mixed with image counts, label-file VINs never validated. |
| L43 | `prepare_finetune_data.py:340,37` | `total_source` counted post-filter; sys.path insert adds `src/vin_ocr/` which cannot make `src.` imports work. |
| L44 | `scripts/prepare_dataset.py:156,114-118` | Keys shuffled without sorting → split not reproducible across machines despite seed; `*.jpg`+`*.JPG` globs double-count on case-insensitive filesystems. |
| L45 | `scripts/compare_models.py:94` | `line.split("\t")` hard-unpack dies on blank/malformed label lines. |
| L46 | `config.py:212-238` | `PipelineConfig.load` restores only 3 sections — `logging`, `vin_length`, `vin_charset` silently dropped on round-trip; tuples come back as lists. |
| L47 | `src/vin_ocr/utils/gpu_utils.py:156,191,157-195` | "Available memory" = total − this process's allocations (overstates on shared GPUs); `hasattr(torch,'hip')` never true (real signal `torch.version.hip`); bare excepts. |
| L48 | `train_vin_model.py:246-287,80-85` | `--model all` silently ignores `--resume/--config/--full/--lora`; falsy-but-legit overrides (0) treated as absent. |
| L49 | `run_experiment.py:516-518,427-447` | Position report sorted lexicographically (`position_1, position_10, …`); "Overall Weighted Metrics" averages train+val+test into one headline number. |
| L50 | `utils/prepare_dataset.py:197 vs prepare_finetune_data.py:85` | Dataset-config metadata records `[3,32,320]` while finetune data documents height 48 — contradictory metadata sources. |

## 5. Verified-clean areas (sweep 1)

- ISO 3779 checksum: weights `(8,7,6,5,4,3,2,10,0,9,8,7,6,5,4,3,2)`, transliteration table,
  index-8 skip, mod-11/`X` — correct (validated against known-valid VINs).
- `core/vin_decode.py` — prefix beam search (Hannun recurrences), checksum gating/repair:
  correct; functionally tested.
- CTC greedy decode (`core/charset.py`) — collapse-then-strip order correct; blank=0
  enforced at load.
- `core/char_metrics.py` alignment→TP/FP/FN mapping, conservation invariants, macro-F1
  support-0 exclusion, corpus CER — correct.
- Both Levenshtein implementations (DP init/recurrence) — correct.
- `src/vin_ocr/utils/prepare_dataset.py` — VIN-grouped split, ratio asserts,
  `assert_no_vin_leakage` — correct.
- `evaluate.py` headline metrics (exact-match/F1/CER denominators, error-row exclusion,
  strict zips) — correct.
- `tracking/run.py`, `tracking/git_state.py`, `tracking/reproduce.py`,
  `evaluation/cli.py`, `training/torch_rosetta.py`, `training/metrics.py` — no logic
  errors found.
- Normalization/BGR conventions and right-side CTC padding in inference paths — correct.
- `config.py` env-bool parsing ("false"→False), split ratios, 33-char VIN charset —
  correct.
- Optuna core search logic (direction=maximize matches objective, bounds sane, all
  suggested params genuinely consumed) — correct apart from M25/L35/L36.

---

# Sweep 1b — modules not covered by the first pass

Covers `src/vin_ocr/web/`, `training/finetune_deepseek.py`, `training/export_onnx.py`,
`training/export_deepseek_onnx.py`, `cli.py`, remaining `tracking/` modules, and the ONNX
conversion scripts. Same severity scale. Key items verified directly against source.

## W-CRITICAL

### W-C1. `src/vin_ocr/web/training_components.py:496-534` — web PaddleOCR fine-tuning can never start
The runner builds `cmd` with `--epochs`, `--batch-size`, `--lr`, `--output-dir`,
`--cpu`/`--gpu`, `--architecture`, `--train-data-dir`, `--train-labels`, `--val-data-dir`,
`--val-labels` — none of which exist in `finetune_paddleocr.py`'s parser (3085–3113: only
`--config/--resume/--export-onnx` + optional DagsHub flags). argparse exits code 2 on every
launch; the UI's "Start Fine-Tuning" always fails. Same defect class as sweep-1 H7. Even if
tolerated, none of the UI's settings could reach the trainer (config YAML is its single
input), so the UI output dir and the trainer's `save_model_dir` diverge by design. The
DeepSeek and Optuna command lines in the same file are correct.

## W-HIGH

### W-H1. `web/app.py:1610,1670` — "Export Results to CSV" unreachable (nested buttons)
The export `st.button` sits inside the `Run Evaluation` button's body; the export click
triggers a rerun in which the outer button is `False` — the export branch can never execute.

### W-H2. `web/app.py:593-612` — Base PP-OCRv3/v4/v5 path uses the 2.x API against pinned paddleocr 3.3.3
`PaddleOCR(..., show_log=False)` (removed in 3.x → init raises), `ocr.ocr(path, cls=True)`
(removed), and `line[1][0]` tuple parsing (3.x returns dict-based results). With
`requirements.txt:24` = `paddleocr==3.3.3`, every "Base" model evaluation lands in the broad
`except` and reports failure.

### W-H3. `scripts/convert_all_to_onnx.py:98-122` — fallback can export RANDOM weights as a converted model
When only `inference.pdiparams` (static/jit format, graph-internal key names) exists, the
script loads it into a fresh dygraph `VINRecognitionModel` via `set_state_dict`, which warns
on unmatched keys but does not raise → the randomly-initialized model is jit-saved, converted
to ONNX, and reported "✅ Converted". Also assumes `paddle.jit.save` produced `.pdmodel`;
PIR-era Paddle writes `.json` (sibling script `reexport_and_convert_onnx.py:100-108` handles
this; this one doesn't), and the temp-file cleanup list omits `.json`.

### W-H4. `training/finetune_deepseek.py:505-515,608-610` — eval metric computed on misaligned sequences, drives best-model selection
`compute_metrics` argmax-decodes causal-LM logits over the full (image+prompt, shifted-by-one)
sequence and compares against the standalone 32-token label encoding — never aligned, no -100
masking. `accuracy` is meaningless, yet `metric_for_best_model="accuracy"`,
`load_best_model_at_end=True`, and `EarlyStoppingCallback` all key off it.

## W-MEDIUM

| ID | Location | Problem |
|----|----------|---------|
| W-M1 | `web/app.py:60-64` | `from ...vin_utils import extract_vin_from_filename, is_valid_vin` — `is_valid_vin` does not exist (only `_is_valid_vin_chars`/`validate_vin`) → ImportError discards BOTH imports; a naive first-17-valid-chars regex replaces the pattern-prioritized extractor. A `20250101123456789-VIN-SAL....jpg` filename yields the timestamp as ground truth. |
| W-M2 | `training_components.py:990-1011 vs 1108` | Race: `stop()` sets `self._process = None` while the monitor thread still dereferences it → AttributeError → `tracker.error()` overwrites the "stopped" status; final UI state is timing-dependent. |
| W-M3 | `training_components.py:990-1011,351-376` | After a server restart, `is_running` is True from the lock/PID of another process, but `stop()` only terminates `self._process` (None) — then deletes the OTHER process's lock and reports success while training keeps running. |
| W-M4 | `web/app.py:773-781` | "Include subdirectories" checkbox changes only the *displayed count* (recursive glob); the actual pool load is always non-recursive with a different extension set. |
| W-M5 | `web/app.py:1908-1911` | `return` inside the Compare tab (when <2 runs) exits the whole dashboard renderer before `with tab_export:` — Export tab permanently empty with one run. |
| W-M6 | `web/app.py:1001-1003` | "Go to Data Management →" writes `session_state.current_page`, which nothing reads (sidebar radio has no key) — button navigates nowhere. |
| W-M7 | `training/export_onnx.py:70,184,274` | Self-test/metadata input contract wrong: channels hardcoded 1, height defaults 32 vs the project contract [N,3,48,320] → post-export "inference test" always fails against repo models; metadata records the wrong shape. `cli.py export` inherits it. |
| W-M8 | `scripts/reexport_and_convert_onnx.py:88-90,174` | Exports with batch dim pinned to 1 (breaks `recognize_batch` contract) and converts `latest.pdparams` (last epoch), not the best checkpoint. |
| W-M9 | `training/export_deepseek_onnx.py:196-228,308` | `torch.onnx.export` binds example inputs POSITIONALLY: the image tensor binds to `input_ids`/`attention_mask`, never `pixel_values` (`input_names` only renames). Self-test feeds float32 into a float16-exported graph — cannot pass. |
| W-M10 | `scripts/fix_inference_models.py:72` | Writes CWD-relative `character_dict_path: ./vin_dict.txt` into inference.yml — resolves against the consumer's CWD, not the yml's dir; "✅ USABLE with PaddleOCR v5 API" verdict is wrong for repo-trained custom-architecture checkpoints. |
| W-M11 | `web/app.py:1322-1326` + `training_components.py:1011` | User-aborted runs display "✅ Training completed!" (`stop()` ends with `tracker.complete(...)`, no error set). For DeepSeek the "(checkpoint saved)" claim is false — no SIGTERM handler exists. |
| W-M12 | `finetune_deepseek.py:291,614,620-633` | `text_labels` forwarded into `model(**batch)` (`remove_unused_columns=False`, collator preserves keys) → TypeError for any forward without `**kwargs`; labels are max_length-padded with no -100 masking (pad tokens contribute to loss). |
| W-M13 | `web/training_components.py:204` | Class-level `_lock_file = Path("./output/.training_lock")` is CWD-relative while everything else resolves against PROJECT_ROOT — the training mutex silently fails when the server is launched from elsewhere. |

## W-LOW

| ID | Location | Problem |
|----|----------|---------|
| W-L1 | `web/app.py:2098-2103` | "Available Models" captions render dict reprs (values are `{"path":..., "type":...}` now). |
| W-L2 | `web/app.py:330-337` | Globbing `*ext` + `*EXT` double-counts every file on case-insensitive filesystems; pool has no intra-batch dedup. |
| W-L3 | `finetune_deepseek.py:33-43 vs 780-841` | Docstring advertises `--gradient-checkpointing/--bf16/--load-in-8bit`; argparse defines none — documented invocations exit 2. |
| W-L4 | `training_components.py:286-288` | `valid_devices` computed, never used (dead device check). |
| W-L5 | `finetune_deepseek.py:136-137` | "Force unbuffered output" loop iterates the module logger's (empty) handler list; basicConfig handlers live on root — no-op. |
| W-L6 | `cli.py:112-114` | `serve` ignores streamlit's return code, always exits 0. |
| W-L7 | `cli.py:193` | `export` help documents a `.pdparams` invocation that `export_paddle_to_onnx` explicitly raises NotImplementedError for. |
| W-L8 | `export_deepseek_onnx.py:128-139` | Tokenizer-fallback failure path leaves `self.tokenizer` unassigned and continues silently. |
| W-L9 | `scripts/generate_system_maps.py:185-190` | Tautological guards (`name not in uses - {name}` always true; `... or True` always true) — only the inner test does anything; docstring-promised entry-point reachability (`ENTRY_POINTS`) never used. |
| W-L10 | `web/app.py:1018,2584-2585` | Dead stores: `state`; System Health computes `gpu_info`/`gpu_available` and never renders them. |
| W-L11 | `scripts/convert_all_to_onnx.py:179-182` | `batch` from input metadata computed then discarded by the shape rebuild. |
| W-L12 | `training_components.py:373` | `except: pass` around status parsing hides malformed-state bugs. |

**Sweep-1b clean files:** `tracking/provenance.py`, `tracking/environment.py`,
`tracking/tracing.py`, `scripts/onnx_example.py` (fallback CTC decode correctly implements
the blank=0 convention), `scripts/train_smoke_test.py` (all replace targets/regexes verified
against the real config and trainer output).

---

# Sweep 2 — Deterministic AST analysis

Full machine results, triage tables, false-positive analysis, and the systemic-class
inventories (30 `zip()` truncation sites, 25 `sys.path` edits, 24 silent swallows, 6
unchecked `os.system` calls, 14 `-O`-stripped asserts) are in `docs/LOGIC_AUDIT_AST.md`.

Sweep-2 verdict in brief: three engines (custom 20-rule AST analyzer, ruff logic rules, MCP
AST audit) independently re-detected sweep-1 findings H8, L8, M27, L40 and the zip/sys.path
defect classes; produced two false positives (both in files sweep 1 had verified clean); and
surfaced no counter-evidence against any documented finding. New sweep-2 items: A1
(`ocr_providers.py:1006` un-importable string annotation), A2 (dead supervision-source stores
in `finetune_paddleocr.py:1833,2600`), A3 (`web/app.py:2584` GPU info computed, never
rendered), plus the systemic inventories above.
