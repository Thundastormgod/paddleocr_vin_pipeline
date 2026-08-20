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

Measured 2026-08-20 in experiment `model_comparison` (one run per model,
identical splits, canonical metrics, every number measured in the run that
reports it). Basis: single-image inference; scratch checkpoints under the
legacy semantics they were trained with (see the 2026-08-20 entry).

| Model | val-102 char / exact | test-40 char / exact |
|---|---|---|
| **Rosetta-ResNet34-IN1K (torch/MPS; production candidate)** | **98.73% / 85.3%** | **99.26% / 90.0%** |
| Rosetta-ResNet34-IN1K + postproc | 95.73% / 85.3% | 98.38% / 90.0% |
| **PP-OCRv3-mobile + pipeline (deployed production default)** | 82.70% / 25.5% | 87.79% / 37.5% |
| PP-OCRv3-mobile + pipeline, postprocessor OFF | 77.51% / 8.8% | 81.03% / 5.0% |
| LCNetV3-SVTR-CTC-ep44 +postproc | 68.28% / 0% | 68.38% / 0% |
| LCNetV3-SVTR-CTC-ep44 (registry v3, alias `best`) | 66.61% / 0% | 68.53% / 0% |
| LCNetV3-SVTR-CTC-ep19 best-val-loss (v2) | 64.13% / 0% | 66.91% / 0% |
| LCNetV3-SVTR-CTC-ep25 (v1) | 59.98% / 0% | 60.88% / 0% |
| Rosetta-ResNet34vd (paddle, scratch) | 63.15% / 0% | 64.56% / 0% |
| LCNetV3-SVTR-CTC-ep36 warm-restart (refuted) | 38.06% / 0% | 42.94% / 0% |

Model naming (2026-08-20): names state what the artifacts ARE.
`LCNetV3-SVTR-CTC` = the custom model's real composition (PPLCNetV3-style
backbone -> SVTR encoder -> CTC head; 3,226,498 params measured; 48x320
input), versioned by training epoch - its class had falsely described
itself as "EXACT PP-OCRv4" (docstring corrected). `PP-OCRv3-mobile` = the
official zoo pair PP-OCRv3_mobile_det + en_PP-OCRv3_mobile_rec. Registry:
`vin-recognizer` -> `vin-recognizer-scratch` -> **`vin-lcnetv3-svtr-ctc`**
(alias `best` = v3/ep44); run names updated in place, metrics untouched.

The postprocessor alone is worth +5.2pp char / +16.7pp exact (val) and
+6.8pp / +32.5pp (test) to the stock engine. Repeat-eval noise floor on
these checkpoints is ~0.1pp char (CPU-thread nondeterminism: v2 re-measured
64.13% vs 64.24% at registration).

The historical Optuna headline - 41.86% exact (18/43), 90.15% char - is no
longer a baseline for anything: it was produced under the early-stop
guillotine (every trial truncated at ~patience+1 epochs), scored through
since-replaced metric implementations on a 43-image corpus under a leaky
split policy, and the fixed tuner showed crashed trials could inherit a
neighbour's score. Its trial corpus was removed from the repo on 2026-08-20
(git history only).

**Fabrication-era documents: removed (2026-08-20).** Every document that
carried unmeasured or extrapolated numbers - the simulated architecture
"benchmark" and its summary JSON, the hardcoded-metrics pipeline script, the
pre-audit results corpus (evaluations made with the wrong-blank decoder and
n=20 extrapolations presented as /382), the truncated-training Optuna trial
corpus, and the fabrication-era summary/status/improvement documents - was
deleted from the working tree in one sweep (90 files). Git history retains
them; nothing in this repository cites them as evidence any more. The rule
survives the cleanup: a number without a run ID is not a measurement.

---

## Entries

### 2026-08-20 — Apple-GPU training stack: warm-started Rosetta hits 90% exact on test

**Hypothesis.** Long-term training must run on this Apple machine; paddle
has no Metal backend and its CPU build is pinned at ~1.1 of 8 cores
(measured: 4.82/4.40/4.41 s/step at OMP 1/4/8). A torch/MPS port plus the
#1 measured lever (pretrained warm start) should finally produce a custom
model that beats the stock engine.

**Change.** New training stack: `torch_rosetta.py` (torchvision ResNet-34,
ImageNet-1k weights, height-only stride surgery, per-column CTC head - no
sequence module, batch-independent by construction) +
`finetune_torch.py` (MPS loop importing the SAME single-source pieces:
VINRecognitionDataset preprocessing, ctc_input_lengths,
update_early_stopping, canonical decode/metrics, tracked runs, MLflow 3
LoggedModel workflow on completion). MPS facts measured first:
`aten::_ctc_loss` unimplemented on MPS -> CPU-bridge loss (log-probs to
CPU, autograd bridges devices); train step batch-16: MPS 0.264s vs
torch-CPU 2.451s (9.3x) vs paddle-CPU 4.40s (16.7x). Cross-stack CTC
parity pinned (torch vs paddle per-sample losses, rtol 1e-3). 14 new
tests.

**Runs.** `train/Rosetta-ResNet34-torch-in1k` (`3e340663`, curves +
finals; post-training pt2 export crashed on the then-missing
input_example - fixed in code - so the LoggedModel + registry step was
completed by `register/Rosetta-ResNet34-IN1K`, metrics measured in-run).
Comparison rows in `model_comparison`. Registry:
`vin-rosetta-resnet34-torch` v1, alias `production-candidate`.

**Result (canonical, single-image).**

| split | char accuracy | exact match |
|---|---|---|
| val-102 | **98.73%** | **85.3%** (87/102) |
| test-40 | **99.26%** | **90.0%** (36/40) |

30 epochs in **52 minutes** on MPS (~100-125s/epoch incl. validation) vs
the paddle-CPU Rosetta's 3.75h for 63.15%/0%. Stock engine surpassed at
epoch 7 (char) and epoch 12 (exact). Wilson 95% CI on 36/40 is
[0.77, 0.96]: consistent with but not yet proof of the ~95% industry
target - a larger held-out set is the next measurement.

**Also measured.**
- Paddle-Rosetta (scratch, same data): val 63.15%/0, test 64.56%/0 - the
  fabricated "46.51% exact" claim is now empirically bounded: the real
  architecture from scratch achieves zero exact matches.
- The postprocessor HURTS this model (val char 98.73 -> 95.73, exact
  unchanged): at this accuracy its extraction/correction can only damage
  already-correct reads. Deployment should use it for checksum GATING
  only, not correction.
- ImageNet download corrupted in-flight once (hash mismatch crash);
  manual fetch verified sha256 b627a593 and cached.

**Verdict.** The from-scratch era is closed twice over. Production
candidate registered; remaining before it becomes the production DEFAULT:
torch inference integration into VINOCRPipeline (the deployed pipeline is
paddle-based), plus a larger test set for the 95% claim. Training on this
machine is now 16.7x faster than the paddle-CPU baseline it replaces.


### 2026-08-20 — Rosetta + ResNet34_vd: the fiction is now a real, measured candidate

**Context.** Fabrication-era documents (removed earlier today) DESCRIBED a
"Rosetta + ResNet34_vd" scoring 46.51% exact match while no such
architecture existed anywhere in code. Ordered follow-up: implement it for
real as a comparison candidate. Nothing carries over from those numbers.

**Change.** `RosettaRecognitionModel` - faithful to Borisyuk, Gordo &
Sivakumar (KDD 2018): ResNet34-vd backbone (deep 3-conv stem, [3,4,6,3]
BasicBlockVd stages, avg-pool downsample shortcuts, height-only strides in
late stages) -> height pool -> per-column Linear -> CTC. NO sequence
module - every timestep predicted from its receptive field alone, so the
model is batch-independent by construction. 21,338,498 params (measured;
6.6x LCNetV3-SVTR-CTC). Geometry: 48x320 -> [B, 512, 3, 80] -> T=80 =
width/4, same timestep axis as the SVTR path and above the 2*17+1 CTC
bound. Wired through trainer dispatch (`algorithm: Rosetta`), registry
loader (rejects legacy mode: this class postdates the batch-axis fix),
`configs/vin_rosetta_config.yml`, and two comparison specs (bare and
+postproc) that skip with a reason until a checkpoint exists. Six new
invariant tests (geometry, full-model batch independence WITH verified
discriminating power - residual paths keep random-init features alive,
unlike the LCNet backbone - parameter budget, no-sequence-module,
both dispatches).

**Measured before launch.** LR sensitivity on the synthetic micro-set:
at the stage-1 recipe's 1e-3 peak the 21.3M ResNet DIVERGED
(13.23 -> 19.21); 3e-4 marginal; 1e-4 fell monotonically (13.23 -> 9.10
over 6 epochs). Real run launched at peak 2e-4, warmup 5, cosine 30 -
the one recorded recipe deviation from stage-1, chosen from measurement.

**Runs.** `train/Rosetta-ResNet34vd-stage1` (`0c1e13dc...`), 2,387-crop
train split, CPU, ~7.5 min/epoch, in progress at entry time. Comparison
numbers will be added to `model_comparison` ONLY when measured; this entry
records the implementation, not a result.


### 2026-08-20 — Batch-axis attention: the metrics were measuring batch composition

**Hypothesis (audit trigger).** The trainer recorded 75.49% val char
accuracy for stage-3b epoch-44 while the registry recorded 66.61% for the
same checkpoint, same split, same canonical metrics - both cannot be the
model's score.

**Investigation (all executed).** Diffing the two eval paths per sample:
97/102 predictions differed between batch-16 and batch-1 on identical
weights. Companion test: one image's logits moved by max|diff| 5.06 when
its batch companions changed, 7.43 vs alone - in eval mode. Layer
bisection with forward hooks: backbone clean, first leak at the neck.
Cause: paddle's `TransformerEncoder` is **batch-first** (`[batch, seq,
dim]` per its own docstring); both encoders transposed to `[T, B, C]` -
the PyTorch convention - so self-attention ran across the batch axis at
every timestep. At batch-1 the encoder degenerated to a per-timestep MLP
(no sequence modeling at all).

**Change (commit f198c8b).** Encoders now feed `[B, T, C]`. Fixed forward
measured batch-independent (max|diff| 0.0; pinned at the neck level by
`tests/test_model_batch_independence.py` - full-model tests cannot
discriminate because an untrained backbone collapses inputs to ~1e-18
features, itself measured). Checkpoints trained under the defect are only
meaningful under it: the same epoch-44 weights score **0.1113** val char
accuracy under the fixed forward. They stay executable via
`legacy_batch_axis_attention=True`, which reproduces the registry number
**exactly (0.6661, all four decimals)**. Semantics travel with artifacts
(semantics.json in logged models; absence = legacy; registry version tags).
ONNX re-export tools had a 130-line drifted inline copy of the
architecture carrying the same defect - replaced with the canonical import
under legacy semantics.

**Runs.** Experiment `model_comparison`, one run per model, identical
splits and metric keys, every number measured in-run (2026-08-20). Table
now lives in "Current measured baseline" above. Key facts: stock engine
margins confirmed on the uniform basis; the postprocessor contributes
+16.7pp (val) / +32.5pp (test) exact match to the stock engine; scratch
stage-3b + postprocessor gains +1.7pp char on val but -0.15pp on test;
repeat-eval noise floor ~0.1pp char (CPU threading).

**Also in this entry (registry/run hygiene).** `vin-recognizer` renamed to
`vin-recognizer-scratch` (family-accurate; alias `scratch-best` -> v3);
version tags record semantics + measurement basis; anonymous
`finetune-PP-OCRv4` runs renamed to what they were: `train/stage1-scratch`,
`train/stage2-scratch-warmrestart-REFUTED`, `train/stage3a-scratch-aborted`,
`train/stage3b-scratch-final`, `probe/{tracked,deepfix,charsel}-micro-6ep`
(none of them were PP-OCRv4, and none were fine-tunes). The ad-hoc
head-to-head run is tagged superseded by `model_comparison`.

**Verdict.** Every number in this logbook now carries its measurement
basis; batched and single-image evaluation are provably identical for all
future models (tested invariant); historical numbers remain reproducible
under their recorded semantics. The from-scratch route stays refuted -
now on a uniform basis.


### 2026-08-19 — REFUTED: checksum-constrained decoding at the current error rate

**Hypothesis.** Position 9 is deterministic (read at 8%, computable at
100%) and confusable glyphs (6<->5, 8<->6/0, B->A) are near-ties - so
CTC prefix beam search + ISO-3779-constrained selection should convert
near-misses into exact matches without retraining.

**Runs.** Decode-only comparison on identical stage-3b epoch-44 weights,
val-102 + test-40 (module: core/vin_decode.py, 10 golden tests).
(Basis note, 2026-08-20: all numbers in this entry were single-image
evaluations - unaffected by the batch-axis attention defect.)

**Result.** Refuted, twice. Ungated constrained selection: val char
accuracy 0.6661 -> 0.6436 (worse). Gated to edit distance <=1 from the
top hypothesis: 0.6661 -> 0.6471 (still worse); test-40 mirrored it
(0.6853 -> 0.6706). Diagnosis: the checksum informs about distance from
the TRUTH; at the current mean ~5 errors/plate no beam hypothesis is
near the truth, while ~9% of arbitrary 17-char strings validate - so
checksum-driven substitutions are pure noise, and position-9 repair
computes a digit from 16 wrong-ish characters.

**Verdict.** The decoder is algorithmically sound (goldens prove the
mechanics: it recovers seeded confusions and computes position 9
correctly when the rest is right) but USELESS BELOW d<=2 - it is kept,
documented with this limit, and wired into no default path. The
sequencing is now measured fact: model quality first (GPU/pretrained
warm start/resolution), constrained decoding after the histogram mass
reaches d<=2. Checkpoint selection by val char accuracy lands with this
entry (loss/accuracy decoupling was measured on stage-3b).

### 2026-08-19 — Stage-3b: 75.5% val char accuracy; val-loss selects the wrong checkpoint

**Hypothesis.** Warm start from stage-1 at peak 3e-4 (evidence-based cap)
continues the descent that 1e-3 destroyed.

**Runs.** MLflow `7bac56fa8a8c443eb710fddb50e3a57a`; registry versions
v2 (epoch-19 best-val-loss) and v3 (epoch-44 final).

**Result.** Confirmed. Early-stopped at epoch 44 (loss-aware, honest:
25 epochs past the epoch-19 val-loss best of 0.7782). Final val
(n=102): **char accuracy 75.49%, F1 0.786**, exact 0/102 - versus
stage-1's 52.19% (positional) / 59.98% (canonical, same weights).
Held-out TEST (n=40, VIN-disjoint): epoch-44 weights **68.53% char /
F1 0.7491**, beating the epoch-19 best-val-loss checkpoint (66.91% /
0.7357) - **validation LOSS kept selecting epoch 19 while char accuracy
kept improving through 44**. Loss/accuracy decoupling is now a measured
fact of this setup; checkpoint selection should track val char accuracy
directly (open item).

> **[BASIS CORRECTION 2026-08-20.]** The 75.49%/0.786 figures came from the
> trainer's BATCHED eval, which - under the batch-axis attention defect
> found later - measured a different function than deployment: attention
> mixed the 16 batch companions into each prediction. The deployment-honest
> (single-image) score of the same epoch-44 weights is **66.61% char /
> 0.7452 F1** (val-102), re-measured and reproduced exactly in
> `model_comparison`. The TEST numbers in this entry (68.53 / 66.91) were
> already single-image and stand as written; the loss-vs-char-accuracy
> decoupling conclusion survives under the corrected basis. The stage-1
> comparison mixes bases: 52.19 was trainer-positional-batched, 59.98 is
> canonical single-image.

**Also observed at run end.** jit.save failed AFTER writing
inference.json; the weights-only fallback then overwrote pdiparams,
producing a mixed-format directory that failed deserialization - caught
by the inference test the moment the graph file appeared. Fallback now
removes partial jit output first (fixed in 73b96a4).

**Verdict.** Third confirmation of the continuation schedule; the
per-char ceiling is now the rare-pattern-family tail and CPU epoch
budget. Next lever: GPU + rare-family oversampling, and a val-char-acc
selection metric.

### 2026-08-19 — Emission anchoring: why padding masks break warm starts

**Hypothesis.** Per-sample CTC input lengths (mask the padded timesteps)
are a strictly better training objective - the padded tail garbage
observed at n=1 suggests the model wastes emissions there.

**Result.** Refuted for warm starts, by direct measurement. The legacy
checkpoint decodes a correct-format VIN yet places its emissions at
timesteps [0,2,3,59..79] of 80 - 10 of 17 characters INSIDE the padded
region (valid_T=64 for that sample). Full-T-trained CTC models anchor
emissions anywhere; the transformer neck owes nothing to visual columns.
Under the masked objective the same 201/201-loaded weights scored
cold-start loss (13.18 vs 0.69 unmasked): the mask cuts off the very
timesteps the model emits in, so no alignment exists.

**Change.** `Global.ctc_mask_padding` flag, default FALSE: full-T
supervision is the simpler-correct objective for fixed-width inputs
(padding learns blanks - CTC's native mechanism - and the inference
contract stays width-free). The mask remains available for fresh runs
wanting anchored emissions; decode-time masking follows the same flag.
Pinned by decode tests (late-emission visible unmasked, ignored masked).

**Verdict.** Stage-3 relaunched unmasked from the stage-1 checkpoint at
peak 3e-4 (run `7bac56fa8a8c443eb710fddb50e3a57a`): first-batch loss
0.69 - warm start intact.

### 2026-08-19 — Stage-2 continuation: warm start + high re-warmup is a mistake

**Hypothesis.** Warm-starting from the epoch-26 checkpoint with a fresh
120-epoch cosine (peak 1e-3) continues the val-loss descent.

**Runs.** MLflow `914573dda4754abbaad6b272c9460c43`.

**Result.** Refuted. Epochs 1-2 (LR ramping 2e-4 -> 4e-4) improved val
loss 0.853 -> 0.847; the ramp to 1e-3 then knocked the model out of its
basin (val loss 1.2-1.7) and it never returned below the epoch-11 best of
0.8309. The loss-aware stopper (shipped mid-stage-1) worked exactly as
designed: stop at epoch 36 after 25 genuinely non-improving epochs.
Final eval at the perturbed epoch-36 weights: char accuracy 44.52%
(vs 52.19% at stage-1's kill point).
> **[BASIS CORRECTION 2026-08-20.]** Both figures were the trainer's
> batched eval under the batch-axis attention defect (see the 2026-08-20
> entry). Single-image re-measurement of the same stage-2 weights:
> **38.06% val char** (`model_comparison`). The verdict is unchanged -
> the warm restart destroyed the basin under every basis. The run predates the
best_val_loss checkpoint, so the epoch-11 state was not saved
(save_epoch_step=5; nearest artifacts epoch_10/epoch_15).

**Verdict.** Continuation runs need a LOW peak LR: the evidence is the
improvement under 2-4e-4 and the destruction at 1e-3. Stage-3 launched
from the stage-1 checkpoint at peak 3e-4 (warmup 2, cosine 60,
loss-aware stopper, min_epochs 10) - and under the fixed trainer it
saves best-by-val-loss checkpoints and streams per-epoch curves to
MLflow. Also observed: paddle.jit static-graph export failed inside
both long training processes while succeeding in short probes; exports
without the graph file are now labelled INCOMPLETE instead of ✅.

### 2026-08-18 — First tracked training run on the real dataset (n=2,387)

**Hypothesis.** With the trainer fixed (CTC contract, checkpoints) and the
real recognition set staged (2,529 crops from DagsHub
`Thundastormgod/jlr-vin-ocr`, VIN-grouped splits 2,387/102/40, 100%
checksum-valid labels), a CPU run under repo-default hyperparameters
should escape blank collapse and learn.

**Runs.** MLflow `29bccdc235fd47cfa0ddaf0ccb683d4f` (experiment
`vin_finetune`; full provenance; replay via
`python -m src.vin_ocr.tracking.reproduce 29bccdc235fd47cfa0ddaf0ccb683d4f`).

**Result.** Blank collapse escaped by ~epoch 3; train loss 13.89 → 0.79,
val loss 13.6 → 0.862 (still falling). Run was killed at epoch 26 by the
old accuracy-only early-stop guillotine (the fix landed mid-flight; the
running process predated it — a live demonstration of readiness item C3).
Final val metrics (n=102, VIN-disjoint): **exact match 0/102, character
accuracy 52.19%, F1 0.5219**. Predictions are all well-formed
`SAL1A2A??SA60????` Land Rover VINs: the model learned the dataset's
shared structure (~11 of 17 characters are common across plates) but not
yet the discriminating characters.
> **[BASIS CORRECTION 2026-08-20.]** 52.19% was the then-current trainer
> metric (positional), computed on a BATCHED eval under the batch-axis
> attention defect. Canonical single-image score of the same weights:
> **59.98% char / 0.6771 F1** (registry v1, re-confirmed in
> `model_comparison`). The under-trained-but-learning verdict stands.

**Verdict.** Pipeline proven end-to-end (data → tracked training →
checkpoints → honest evaluation); model under-trained at 26 CPU epochs
with LR nearly decayed. Not comparable to the historical 41.86%: that
number came from a 43-image corpus under a leaky split policy, truncated
training AND a broken decoder. Continuation run warm-started from this
checkpoint with the loss-aware stopper; GPU training is the real path to
convergence.

### 2026-08-18 — The trainer could not train: CTC contract + early-stop guillotine

**Hypothesis.** Before attempting an enterprise training run, execute the
trainer end-to-end on a synthetic micro-dataset (12 rendered VIN plates,
valid check digits) and catalogue every failure.

**Result (all reproduced by execution, fixed in `a6ec119`).**
1. The trainer crashed on the first batch under paddle 3.3.x: warpctc
   requires labels int32 + length tensors int64; the code passed
   all-int32. No one had ever run this trainer on paddle 3.3.
2. warpctc receives raw logits (it applies softmax internally); the
   trainer fed log_softmax output.
3. Early stopping was unconditionally fatal: the improvement test
   compared `val > val + min_delta` (best already updated), the patience
   counter never reset, and **every run died after exactly patience+1
   epochs regardless of progress**. Every historical Optuna trial
   (patience 3-20) trained under this guillotine — no historical run was
   allowed past ~21 epochs. The pre-audit trial corpus (removed
   2026-08-20) measures truncated training.
4. `latest` checkpoints: not refreshed every epoch, no resume info
   (resume-from-latest restarted the epoch counter and warmup),
   non-atomic writes.

**Verification.** After fixes: trainer learns (micro-set loss
13.8 → 0.095, 11/12 exact memorization under repo-default
hyperparameters), bit-identical same-seed runs, resume-from-latest
continues at the right epoch with scheduler state, early stopping fires
only without genuine improvement. Also measured: constant lr 3e-3
collapses this architecture into the blank basin permanently (600 steps,
no escape) while 1e-3 learns — the Optuna space reaching 1e-2 samples a
poison region.

**Effect on model metrics.** None yet — but the 41.86% (18/43) baseline
must now be read as "best result achievable under ≤21-epoch truncated
training evaluated through a then-broken decoder". Re-run the study
before comparing anything against it. Full readiness list:
ENTERPRISE_TRAINING_READINESS.md.

### 2026-08-18 — n=1 validation: the fine-tuned checkpoint reads VINs

**Hypothesis.** The recorded 0.0% for
`output/vin_rec_finetune/best_accuracy.pdparams` (pre-audit multi-model
results, 50 images; file removed 2026-08-20) was produced by the
wrong-blank ONNX decoder (blank=33 against a blank=0 model) and measures
the decoder, not the model. If so, decoding the same checkpoint through
the canonical charset should produce VIN-like text, not garbage.

**Change.** None to the model. paddlepaddle 3.3.1 (CPU) installed;
checkpoint loaded into the current `VINRecognitionModel` (201/201 keys,
zero shape mismatches — architecture code and checkpoint agree); the one
image in the repo preprocessed exactly as the training dataset does;
greedy-decoded with `core.charset.ctc_greedy_decode`.

**Runs.** MLflow `66ec508d80344c24bcac028f2a7828ae`
(experiment `checkpoint_validation`; replay:
`python -m src.vin_ocr.tracking.reproduce 66ec508d80344c24bcac028f2a7828ae`).

**Result.** n=1 (`data/1-VIN -SAL1A2A40SA606662.jpg`, GT
`SAL1A2A40SA606662`). Raw greedy decode (T=80):
`SAL1A2A40SA6062942010170736465617626961547607627601201216` — the
prefix is unmistakably the plate being read. First-17 window
`SAL1A2A40SA606294`: edit distance **3/17**, char accuracy **0.8235**,
alignment F1 **0.8824**, exact match **0/1**, mean kept-step probability
0.729.

**Verdict.** Hypothesis supported at n=1: the checkpoint reads most of
the VIN; the 0.0% on record was a decoder artifact. Two real failure
modes observed and now on record: (a) the model emits digit garbage over
the black-padded tail instead of blanks (57 chars decoded from 80
timesteps), so window selection matters; (b)
`extract_vin_from_text` picked a WORSE window than the plain prefix on
this garbage-tailed input (edit distance 10 vs 3) — its window-selection
heuristic deserves a look. A full re-evaluation needs the 381-image
DagsHub set, which is not in the repo (no DVC pointers; external
credentials required). At n=1 nothing beyond "the decoder was the
problem" is claimed.

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
- Historical multi-model evaluation results (file removed 2026-08-20) are
  NOT comparable to new runs: the old character metrics were positional and
  '_'-padded, and every old ONNX row was decoded with the wrong blank index.
  In particular, **the recorded 0.0% for `output/vin_rec_finetune` measured
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
not what is computed. The then-standing 41.86% (18/43) baseline came from the
root tuner's pre-audit trial corpus (removed 2026-08-20); any trial in it
whose training crashed may carry a neighbour's accuracy under its own
hyperparameters, so per-trial hyperparameter conclusions drawn from it were
suspect and no number from it is citable.

### 2026-08-17 — Experiment tracking with mandatory provenance

**Problem.** Three consecutive audits found metrics that could not be traced to
the code that produced them: a hardcoded 46.51%, `# Simulated` training results
returned without calling `train()`, and sample sizes extrapolated by a factor
of 19. The common cause was structural, not careless — repo-wide, **zero** files
recorded a commit SHA alongside a metric, so no result could be checked against
the code that produced it. The 63 files of the pre-audit trial corpus (removed 2026-08-20) carried
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

### 2026-08-19 — Head-to-head: stock PaddleOCR pipeline vs fine-tuned v3

**Hypothesis.** After the full dependency install, the stock pretrained engine
(PP-OCRv3 mobile det+rec inside `VINOCRPipeline`, engraved preprocessing,
postprocessor on) scored an exact match at 0.98 confidence on the bundled
plate; if that generalises, the from-scratch fine-tune is not the production
path.

**Change.** No code change — a measurement. Same val-102/test-40 splits, same
canonical `char_level_metrics`, prediction = pipeline `vin` field.

**Runs.** `15525c804771492c91a0b47bd6a83067` (head2head-stock-pipeline) vs
stage-3b `7bac56fa` (registry `vin-recognizer` v3).

**Result.**

| metric | stock, val-102 | v3, val-102 | stock, test-40 | v3, test-40 |
|---|---|---|---|---|
| exact match | **25.5%** (26) | 0% | **37.5%** (15) | 0% |
| char accuracy | **82.70%** | 66.61% | **87.79%** | 68.53% |
| F1 (micro) | **0.857** | 0.7452 | **0.905** | 0.749 |
| CER | **0.173** | 0.334 | **0.122** | 0.315 |

> **[CORRECTION 2026-08-20.]** The v3 val F1 originally read "~0.69" - an
> estimate written where a measured value existed (registry run
> `c689e775`: 0.7452, re-confirmed in `model_comparison`). Replaced. All
> other cells were measured. Both columns are single-image basis, so the
> comparison is deployment-honest; margins re-confirmed on the uniform
> basis in `model_comparison` (2026-08-20).

Throughput ~0.18 s/image on CPU (dedup cache off, n=142, errors=0).

**Verdict.** Decisive: the stock pretrained engine beats the from-scratch
fine-tune by +16-19 points char accuracy and 26/142 → 41/142 exact matches vs
zero. Training from random init on 2,387 crops was never competitive with
weights pretrained on millions of text lines — improvement-ladder #2
(pretrained warm start) is hereby promoted from "next candidate" to the only
sanctioned training path. Practical consequences: (1) production default
today is the stock engine + this repo's pre/post-processing; (2) all future
fine-tunes start from `en_PP-OCRv3` rec weights and must beat THIS baseline
(82.70% val char, 25.5% val exact), not the from-scratch numbers; (3) the
from-scratch v1-v3 registry entries remain as the honest record of that
refuted route.

### Template for the next entry

```
### YYYY-MM-DD — <one-line summary>

**Hypothesis.** <what you expect to change, and the mechanism>
**Change.** <what you actually changed>
**Runs.** <MLflow run IDs>
**Result.** <measurement with sample size, vs the baseline above>
**Verdict.** <kept / reverted / inconclusive at this n>
```
