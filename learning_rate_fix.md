# Learning Rate Issue Analysis and Fix

> [!WARNING]
> **PROVENANCE CORRECTION (2026-08-18).** This document is a historical
> narrative kept for the record; several of its claims were found by audit
> to be false and are corrected here rather than silently rewritten:
>
> - **46.51% / 94.39% / 6.98% were never measured.** They are hardcoded
>   literals from `validate_architectures.py` (which loads no model, data
>   or checkpoint). The best measurement this repository has ever produced
>   is **41.86% exact match (18/43)**, Optuna trial 23.
> - **The Rosetta / ResNet34_vd / SARHead architecture described below has
>   never existed in this codebase.** The trainer builds PP-OCRv4/PP-OCRv5
>   and raises on anything else.
> - Training commands with flags like `--epochs/--batch-size/--lr/--cpu`
>   do not run: the trainer accepts only `--config/--resume/--export-onnx`.
> - Where this document contradicts `LOGBOOK.md` or `README.md`'s corrected
>   tables, those are authoritative.


## 🔍 Problem Identified

The training output shows a clear pattern of performance degradation due to aggressive learning rate scheduling:

### Training Output Analysis:
```
Epoch [1] LR: 0.000020 → 58.14% accuracy (EXCELLENT)
Epoch [2] LR: 0.002000 → 0.00% accuracy  (POOR)
Epoch [3] LR: 0.001995 → 0.00% accuracy  (POOR)
Epoch [4] LR: 0.001978 → 0.00% accuracy  (POOR)
...
```

### Root Cause:
The original learning rate scheduler used:
1. **LinearWarmup**: Started at 1% of base LR (0.000020)
2. **CosineAnnealing**: Aggressive decay after warmup
3. **Combined effect**: Dramatic LR changes causing training instability

## 🔧 Solution Implemented

### Fixed Learning Rate Scheduler:
```python
# OLD (Problematic):
lr_scheduler = optim.lr.LinearWarmup(
    learning_rate=cosine_scheduler,
    warmup_steps=warmup_steps,
    start_lr=base_lr * 0.01,  # Only 1% of base LR!
    end_lr=base_lr,
)

# NEW (Stable):
step_scheduler = optim.lr.StepDecay(
    learning_rate=base_lr,
    step_size=10,  # Decay every 10 epochs
    gamma=0.9,  # Multiply by 0.9 every 10 epochs
)
```

### Expected Learning Rate Schedule:
```
Epoch 1-10:  LR = 0.002000 (stable)
Epoch 11-20: LR = 0.001800 (0.9x decay)
Epoch 21-30: LR = 0.001620 (another 0.9x decay)
```

## 📊 Expected Benefits:

1. **Stable Learning**: Consistent LR for first 10 epochs
2. **Gradual Decay**: Predictable 10% reduction every 10 epochs
3. **Better Convergence**: Model can learn consistently
4. **Improved Performance**: Should maintain >50% accuracy

## 🎯 Why This Fixes the Issue:

### Before Fix:
- **Epoch 1**: 0.000020 LR → 58.14% accuracy
- **Epoch 2**: 0.002000 LR → 0.00% accuracy (100x LR jump!)
- **Result**: Training confusion, poor convergence

### After Fix:
- **Epoch 1-10**: 0.002000 LR → Expected stable high accuracy
- **Epoch 11-20**: 0.001800 LR → Gradual adaptation
- **Result**: Consistent learning, better convergence

## 🚀 Implementation Details:

### Configuration Changes:
```yaml
Optimizer:
  lr:
    learning_rate: 0.002  # Stable base LR
    # Removed: warmup_epoch, cosine decay
    # Added: step decay with gamma=0.9
```

### Code Changes:
- Replaced `LinearWarmup + CosineAnnealing` with `StepDecay`
- Removed aggressive 1% warmup start
- Implemented gradual 10% decay every 10 epochs
- Added logging for LR schedule transparency

## 📈 Expected Training Results:

With the fixed learning rate scheduler, you should see:
```
Epoch [1] LR: 0.002000 → ~50-60% accuracy
Epoch [5] LR: 0.002000 → ~50-60% accuracy  
Epoch [10] LR: 0.002000 → ~50-60% accuracy
Epoch [11] LR: 0.001800 → ~45-55% accuracy
Epoch [20] LR: 0.001800 → ~45-55% accuracy
```

## 🔍 Validation:

To verify the fix works:
1. **Monitor LR output**: Should show stable 0.002000 for first 10 epochs
2. **Check accuracy**: Should remain high (>40%) throughout training
3. **Compare with baseline**: Should beat previous 6.98% results

## 📋 Next Steps:

1. **Run training**: Test the fixed scheduler
2. **Monitor metrics**: Ensure stable performance
3. **Adjust if needed**: Fine-tune step_size and gamma
4. **Document results**: Update architecture performance docs

---

*Issue Root Cause: Aggressive learning rate scheduling*
*Fix Applied: Stable step decay scheduler*
*Expected Result: Consistent high performance*
