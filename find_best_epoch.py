#!/usr/bin/env python3
"""
Find the best epoch with highest accuracy from saved checkpoints
"""

import sys
import traceback
from pathlib import Path

import paddle
import yaml

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

#: Performance thresholds as RATIOS of the actual validation-set size. They
#: preserve the intent of the historical absolute counts (15/43 "good",
#: 22/43 "target") without baking one validation set's size into the script.
GOOD_RATIO = 15 / 43
TARGET_RATIO = 22 / 43


def find_best_epoch():
    """Test all saved epochs to find the best performing one."""

    print("🔍 Finding Best Epoch Performance")
    print("=" * 50)

    # Load config
    with open(project_root / "configs/vin_finetune_config.yml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize trainer
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    trainer = VINFineTuner(config)

    val_total = len(trainer.val_loader.dataset)

    # Find all epoch checkpoints
    checkpoint_dir = project_root / "output/vin_rec_finetune"
    epoch_files = list(checkpoint_dir.glob("epoch_*.pdparams"))
    epoch_files.sort(key=lambda x: int(x.stem.split('_')[1]))

    print(f"📁 Found {len(epoch_files)} epoch checkpoints")
    print(f"📊 Validation set size: {val_total}")

    results = []

    for epoch_file in epoch_files:
        epoch_num = int(epoch_file.stem.split('_')[1])

        try:
            # Load model weights
            trainer.model.set_state_dict(paddle.load(str(epoch_file)))

            # validate() returns a 3-tuple:
            # (val_loss, exact_match_accuracy, char_accuracy)
            val_loss, val_acc, val_char_acc = trainer.validate()

            # Get detailed metrics
            metrics = trainer.get_last_validation_metrics()
            correct = metrics.get('correct_images', 0)
            total = metrics.get('total_images', 0)

            result = {
                'epoch': epoch_num,
                'accuracy': val_acc,
                'correct': correct,
                'total': total,
                'char_accuracy': val_char_acc,
                'loss': val_loss
            }

            results.append(result)

            print(f"Epoch {epoch_num:2d}: {correct:2d}/{total} correct ({val_acc:.4f}) | Char: {val_char_acc:.4f}")

        except Exception:
            # Per-epoch isolation: one unreadable checkpoint must not kill
            # the sweep, but the real error must SURFACE with its traceback,
            # not vanish into a one-line "ERROR" (a tuple-unpack bug hid
            # behind exactly that for every epoch).
            print(f"Epoch {epoch_num:2d}: ERROR")
            traceback.print_exc()

    if not results:
        print(f"\n❌ No epoch could be scored ({len(epoch_files)} checkpoint file(s) found)")
        return None, 0, val_total

    # Best = the earliest epoch achieving the maximum correct count. Chosen
    # over the SCORED results: the old in-loop `correct > best` tracking
    # reported best_epoch=None whenever every epoch scored 0 correct, which
    # is indistinguishable from "nothing could be scored".
    best = max(results, key=lambda r: r['correct'])
    best_epoch = best['epoch']
    best_correct = best['correct']
    best_accuracy = best['accuracy']

    # Summary
    print("\n🎯 BEST PERFORMANCE:")
    print(f"   Epoch: {best_epoch}")
    print(f"   Correct: {best_correct}/{val_total}")
    print(f"   Accuracy: {best_accuracy:.4f}")

    # Thresholds are ratios of the ACTUAL val set, not hardcoded /43 counts.
    good_threshold = GOOD_RATIO * val_total
    target_threshold = TARGET_RATIO * val_total

    print(f"\n📊 HIGH PERFORMANCE EPOCHS (>= {GOOD_RATIO:.0%} of {val_total}):")
    high_perf = [r for r in results if r['correct'] >= good_threshold]
    high_perf.sort(key=lambda x: x['correct'], reverse=True)

    for i, r in enumerate(high_perf[:5], 1):
        print(f"   {i}. Epoch {r['epoch']:2d}: {r['correct']}/{val_total} correct ({r['accuracy']:.4f})")

    if best_correct >= target_threshold:
        print(f"\n🎉 FOUND EPOCH AT >= {TARGET_RATIO:.0%} EXACT MATCH!")
        print(f"   Use epoch {best_epoch} for best performance")

        # Epoch checkpoints carry their own .pdopt and _info.json, so resume
        # from the checkpoint directly instead of copying only the weights
        # over best_accuracy.pdparams (which would strand a stale info file).
        print("\n🚀 To resume from this best epoch:")
        print(f"   python resume_training.py --checkpoint output/vin_rec_finetune/epoch_{best_epoch}")

    elif best_correct >= good_threshold:
        print(f"\n✅ Found good performance epoch: {best_epoch} with {best_correct} correct")

    else:
        print(f"\n⚠️ Best performance is only {best_correct} correct predictions")

    return best_epoch, best_correct, val_total


if __name__ == '__main__':
    best_epoch, best_correct, val_total = find_best_epoch()
    if best_epoch is None:
        print("\n❌ FAILED: no epoch could be scored")
        sys.exit(1)
    print(f"\n🏆 Best epoch: {best_epoch} with {best_correct}/{val_total} correct predictions")
