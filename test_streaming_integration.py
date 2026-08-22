#!/usr/bin/env python3
"""
Test DagsHub Streaming Integration

Smoke-tests the streaming mechanism this repository ACTUALLY uses:

  train_vin_streaming.py -> dagshub.streaming.install_hooks -> VINFineTuner

The previous version of this script tested a phantom module
(src.vin_ocr.data.dagshub_integration) that never existed in this
repository, plus a /mnt FUSE mount that is Linux-only. Both are gone.

What is verified here, with no network access:

1. The dagshub streaming entry points are importable.
2. StreamingVINTrainer re-anchors dataset paths into a NEW *_streaming
   config without clobbering sibling keys or the original file.
3. VINRecognitionDataset reads image bytes through Python's open() -
   the ONLY file API dagshub's install_hooks can intercept - proven by
   serving a deleted file from a patched builtins.open, exactly the way
   the hooks materialize remote-only files. cv2.imread (native fopen)
   would fail this test.
"""

import builtins
import io
import shutil
import sys
import tempfile
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def test_streaming_imports() -> bool:
    """The real streaming stack must be importable."""
    print("🔍 Testing Streaming Imports")
    print("-" * 30)

    try:
        from dagshub.streaming import install_hooks  # noqa: F401
        print("✅ dagshub.streaming.install_hooks importable")
    except ImportError as e:
        print(f"❌ dagshub package missing: {e}")
        return False

    try:
        from train_vin_streaming import (  # noqa: F401
            DagsHubDataStreamer,
            StreamingVINTrainer,
        )
        print("✅ train_vin_streaming entry points importable")
        return True
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False


def test_streaming_config_rewrite() -> bool:
    """_update_config_for_streaming: deep-merge, new file, anchored paths."""
    print("\n📄 Testing Streaming Config Rewrite")
    print("-" * 30)

    try:
        import yaml
        from train_vin_streaming import StreamingVINTrainer

        src_config = project_root / "configs" / "vin_finetune_config.yml"
        with tempfile.TemporaryDirectory() as tmp:
            work_config = Path(tmp) / "vin_finetune_config.yml"
            shutil.copy(src_config, work_config)

            trainer = StreamingVINTrainer(str(work_config), use_streaming=False)
            trainer._update_config_for_streaming()

            streaming_path = Path(trainer.config_path)
            if streaming_path == work_config:
                print("❌ Original config was overwritten")
                return False
            print(f"✅ New config written: {streaming_path.name}")

            with open(streaming_path) as f:
                rewritten = yaml.safe_load(f)

            for section in ("Train", "Eval"):
                data_dir = rewritten[section]["dataset"]["data_dir"]
                if not Path(data_dir).is_absolute():
                    print(f"❌ {section}.dataset.data_dir not anchored: {data_dir}")
                    return False
                if "loader" not in rewritten[section]:
                    print(f"❌ deep-merge lost {section}.loader")
                    return False
            print("✅ Dataset paths anchored, sibling keys survive")
            return True
    except Exception as e:
        print(f"❌ Config rewrite test failed: {type(e).__name__}: {e}")
        return False


def test_dataset_reads_via_python_open() -> bool:
    """Image bytes must flow through builtins.open (hookable), not native I/O.

    Simulates dagshub materialization: the image file is DELETED from disk
    and served from a patched builtins.open. cv2.imread would return None
    here; the dataset must still produce a real tensor.
    """
    print("\n🖼️ Testing Hook-Compatible Image Reads")
    print("-" * 30)

    try:
        import cv2
        import numpy as np
        from src.vin_ocr.core.charset import load_char_dict
        from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

        char_to_idx, _ = load_char_dict("configs/vin_dict.txt")

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            img = np.full((64, 320, 3), 128, np.uint8)
            cv2.putText(img, "SAL1A2A40SA606662", (5, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            img_file = tmp_path / "vin.jpg"
            cv2.imwrite(str(img_file), img)
            (tmp_path / "labels.txt").write_text("vin.jpg\tSAL1A2A40SA606662\n")

            ds = VINRecognitionDataset(
                data_dir=str(tmp_path),
                label_file=str(tmp_path / "labels.txt"),
                char_dict=char_to_idx,
                is_training=False,
            )

            # Remote-only simulation: bytes exist, file does not.
            data = img_file.read_bytes()
            img_file.unlink()

            real_open = builtins.open

            def hooked_open(file, mode="r", *args, **kwargs):
                if str(file) == str(img_file):
                    return io.BytesIO(data)
                return real_open(file, mode, *args, **kwargs)

            builtins.open = hooked_open
            try:
                item = ds[0]
            finally:
                builtins.open = real_open

            if item["image"].shape[0] != 3:
                print(f"❌ Expected CHW tensor, got shape {item['image'].shape}")
                return False
            if item["text"] != "SAL1A2A40SA606662":
                print(f"❌ Wrong label round-trip: {item['text']}")
                return False
            print("✅ Deleted file served through patched open() -> real tensor")
            print("   (cv2.imread's native fopen would have returned None)")
            return True
    except Exception as e:
        print(f"❌ Hooked-read test failed: {type(e).__name__}: {e}")
        return False


def main() -> bool:
    """Run all tests."""
    print("🧪 DagsHub Streaming Integration Test")
    print("=" * 50)

    tests = [
        test_streaming_imports,
        test_streaming_config_rewrite,
        test_dataset_reads_via_python_open,
    ]

    results = []
    for test in tests:
        try:
            results.append(test())
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {type(e).__name__}: {e}")
            results.append(False)

    print("\n📊 Test Results")
    print("=" * 20)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("🎉 All tests passed! Streaming is ready to use.")
        print("\n🚀 Next steps:")
        print("1. Set DagsHub credentials:")
        print("   export DAGSHUB_USERNAME='your_username'")
        print("   export DAGSHUB_TOKEN='your_token'")
        print("2. Run streaming training:")
        print("   python train_vin_streaming.py --stream \\")
        print("     --repo-owner <owner> --repo-name <repo> \\")
        print("     --dagshub-user $DAGSHUB_USERNAME --dagshub-token $DAGSHUB_TOKEN")
    else:
        print("⚠️  Some tests failed. Check the errors above.")

    return passed == total


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
