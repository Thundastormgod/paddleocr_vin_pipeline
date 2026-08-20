"""
Model versioning and traced inference for VIN recognition checkpoints.

MLflow 3.x tailoring for this project (the iris/ElasticNet pattern, made
real for paddle CTC checkpoints):

- A checkpoint becomes a **LoggedModel** entity: a pyfunc wrapping
  checkpoint weights + the canonical charset + the canonical CTC decode,
  so the versioned artifact IS the deployable unit - not a bare state
  dict whose decode contract lives in someone's head. (This repository's
  history includes an evaluator that decoded a correct model into garbage
  because the decode contract was duplicated; the versioned model carries
  its own.)
- Metrics attached to a model version are MEASURED here, at registration
  time, on a named **Dataset** entity built from a label file - never
  transcribed from elsewhere. `mlflow.log_metrics(model_id=...,
  dataset=...)` links model, metrics and data exactly as in the MLflow 3
  LoggedModel workflow.
- Each version is registered under one registry name with tags carrying
  provenance (checkpoint path, source training run, git commit).
- `traced_recognize()` wraps single-image inference in MLflow **spans**
  (preprocess -> forward -> decode -> validate), so the Traces tab shows
  per-stage timing and intermediate outputs for any recognition.

Requires the [tracking] extra plus paddle at load/serve time.
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# This module IS the mlflow integration: a missing mlflow must fail loudly
# at import, not surface later as a half-registered model.
import mlflow
import mlflow.pyfunc

#: Registry name of the custom from-scratch family, named for what the
#: model actually is: PPLCNetV3-style backbone -> SVTR encoder -> CTC head,
#: 3.23M parameters (v1-v3, refuted route - see LOGBOOK 2026-08-19/20).
#: New model families (e.g. pretrained warm starts) must register under
#: their OWN name, passed explicitly.
REGISTERED_MODEL_NAME = "vin-lcnetv3-svtr-ctc"


# =============================================================================
# PYFUNC MODEL - the versioned, deployable unit
# =============================================================================

def _load_recognizer(checkpoint_path: str, config_path: str, dict_path: str,
                     legacy_batch_axis_attention: bool = False):
    """
    Build the recognition model and its decode context from a checkpoint.

    Args:
        legacy_batch_axis_attention: Load under the pre-2026-08-20 defect
            semantics (transformer attends across the batch axis). Required
            for checkpoints TRAINED under that defect: under the fixed
            forward the same weights score 0.1113 val char accuracy vs
            0.6661 under the semantics they were trained with (measured,
            stage-3b epoch 44, val-102). Never use for new checkpoints.

    Returns:
        (model, idx_to_char, config) with the model in eval mode.

    Raises:
        FileNotFoundError: If any input file is missing.
        ValueError: If the checkpoint does not fit the architecture
            (missing/unexpected/mismatched keys) - a wrong-architecture
            checkpoint must fail registration, not serve garbage.
    """
    import paddle
    import yaml

    from src.vin_ocr.core.charset import load_char_dict, num_classes
    from src.vin_ocr.training.finetune_paddleocr import VINRecognitionModel

    for path in (checkpoint_path, config_path):
        if not Path(path).is_file():
            raise FileNotFoundError(f"required file missing: {path}")

    config = yaml.safe_load(Path(config_path).read_text())
    char_to_idx, idx_to_char = load_char_dict(dict_path)

    model = VINRecognitionModel(
        config, num_classes(char_to_idx),
        legacy_batch_axis_attention=legacy_batch_axis_attention,
    )
    state = paddle.load(checkpoint_path)

    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    if model_keys != ckpt_keys:
        raise ValueError(
            f"checkpoint does not fit the architecture: "
            f"{len(model_keys - ckpt_keys)} missing, "
            f"{len(ckpt_keys - model_keys)} unexpected keys"
        )
    model.set_state_dict(state)
    model.eval()
    return model, idx_to_char, config


class VINRecognizerPyfunc(mlflow.pyfunc.PythonModel):
    """
    pyfunc wrapper: CHW float32 tensors in, decoded VINs out.

    predict() input: np.ndarray [B, 3, 48, 320], already preprocessed the
    way training preprocesses (VINRecognitionDataset._preprocess_image).
    Output: list of {'vin', 'confidence', 'checksum_valid'}.
    """

    def load_context(self, context) -> None:
        # Semantics basis: a "semantics.json" artifact records which forward
        # the checkpoint was trained/measured under. Its ABSENCE means the
        # model was logged before the batch-axis attention fix (2026-08-20),
        # so it must load under the legacy semantics it was measured with -
        # under the fixed forward those weights score 0.1113 vs 0.6661 val
        # char accuracy (measured). New registrations write the file.
        legacy = True
        semantics_path = context.artifacts.get("semantics")
        if semantics_path and Path(semantics_path).is_file():
            semantics = json.loads(Path(semantics_path).read_text())
            legacy = bool(semantics["legacy_batch_axis_attention"])
        self.model, self.idx_to_char, self.config = _load_recognizer(
            context.artifacts["checkpoint"],
            context.artifacts["config"],
            context.artifacts["char_dict"],
            legacy_batch_axis_attention=legacy,
        )

    def predict(self, context, model_input, params=None) -> List[Dict[str, Any]]:
        import paddle

        from src.vin_ocr.core.charset import ctc_greedy_decode
        from src.vin_ocr.core.vin_utils import validate_vin

        batch = np.asarray(model_input, dtype=np.float32)
        if batch.ndim == 3:
            batch = batch[None, ...]
        if batch.ndim != 4 or batch.shape[1:] != (3, 48, 320):
            raise ValueError(
                f"expected input [B, 3, 48, 320], got {batch.shape}"
            )

        with paddle.no_grad():
            logits = self.model(paddle.to_tensor(batch))
        probs = paddle.nn.functional.softmax(logits, axis=-1).numpy()

        results = []
        for sample in probs:
            text, kept = ctc_greedy_decode(sample.argmax(-1), self.idx_to_char)
            vin = text[:17]
            confidence = float(sample.max(-1)[kept].mean()) if kept else 0.0
            results.append({
                "vin": vin,
                "confidence": confidence,
                "checksum_valid": validate_vin(vin).checksum_valid,
            })
        return results


# =============================================================================
# MEASURED METRICS - never transcribed
# =============================================================================

def evaluate_checkpoint(
    checkpoint_path: str,
    label_file: str,
    data_dir: str = "finetune_data",
    config_path: str = "configs/vin_finetune_config.yml",
    dict_path: str = "configs/vin_dict.txt",
    max_samples: Optional[int] = None,
    legacy_batch_axis_attention: bool = False,
    postprocess: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a checkpoint on a label file with the canonical metrics.

    Every image is decoded one at a time (batch size 1): the number this
    returns is the deployment-relevant single-image score by construction,
    and under the fixed batch-first forward it is provably identical to any
    batched evaluation (batch independence is a tested model invariant).
    Set legacy_batch_axis_attention=True only for checkpoints trained
    before the 2026-08-20 fix - see _load_recognizer.

    Args:
        postprocess: Score the deployable variant - each decoded string is
            run through VINPostProcessor (artifact stripping, charset
            fixes, VIN extraction) before comparison, exactly as the
            pipeline serves it. Default False scores the bare decode.

    Returns:
        exact_match, char_accuracy, f1_micro, precision, recall, cer,
        checksum_valid_rate, n - all measured here by decoding every image.
    """
    import paddle

    from src.vin_ocr.core.char_metrics import char_level_metrics
    from src.vin_ocr.core.charset import ctc_greedy_decode, load_char_dict
    from src.vin_ocr.core.vin_utils import validate_vin
    from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

    model, idx_to_char, _ = _load_recognizer(
        checkpoint_path, config_path, dict_path,
        legacy_batch_axis_attention=legacy_batch_axis_attention,
    )
    char_to_idx, _ = load_char_dict(dict_path)
    dataset = VINRecognitionDataset(
        data_dir=data_dir, label_file=label_file,
        char_dict=char_to_idx, is_training=False,
    )
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))

    post = None
    if postprocess:
        from src.vin_ocr.pipeline.vin_pipeline import VINPostProcessor
        post = VINPostProcessor()

    pairs, checksum_ok = [], 0
    with paddle.no_grad():
        for i in range(n):
            item = dataset[i]
            logits = model(paddle.to_tensor(item['image'][None])).numpy()[0]
            text, _ = ctc_greedy_decode(logits.argmax(-1), idx_to_char)
            if post is not None:
                pred = post.process(text)["vin"] or ""
            else:
                pred = text[:17]
            pairs.append((pred, item['text']))
            checksum_ok += validate_vin(pred).checksum_valid

    metrics = char_level_metrics(pairs)
    exact = sum(1 for p, g in pairs if p == g)
    return {
        "exact_match": exact / n if n else 0.0,
        "char_accuracy": metrics.char_accuracy,
        "f1_micro": metrics.f1_micro,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "cer": metrics.cer,
        "checksum_valid_rate": checksum_ok / n if n else 0.0,
        "n": float(n),
    }


# =============================================================================
# VERSIONING - LoggedModel + registry
# =============================================================================

def register_checkpoint_version(
    checkpoint_path: str,
    label_file: str = "finetune_data/val_labels.txt",
    registered_name: str = REGISTERED_MODEL_NAME,
    source_run_id: Optional[str] = None,
    stage_label: str = "",
    config_path: str = "configs/vin_finetune_config.yml",
    dict_path: str = "configs/vin_dict.txt",
    experiment: str = "vin_finetune",
    max_eval_samples: Optional[int] = None,
    legacy_batch_axis_attention: bool = False,
) -> Dict[str, Any]:
    """
    Log a checkpoint as an MLflow LoggedModel, attach MEASURED metrics
    linked to a named Dataset entity, and register it as a new version.

    Args:
        checkpoint_path: .pdparams training checkpoint.
        label_file: Dataset the metrics are measured on (linked as an
            mlflow Dataset named after the file).
        registered_name: Registry name; each call creates the next version.
        source_run_id: Training run that produced the checkpoint, recorded
            as a tag so the version points back at its provenance.
        stage_label: Human label ('stage-1', 'stage-3b-best-val-loss', ...).
        max_eval_samples: Cap evaluation size (None = full label file).

    Returns:
        {'model_id', 'model_uri', 'registered_name', 'version', 'metrics'}
    """
    import pandas as pd

    mlflow.set_experiment(experiment)

    # Measure BEFORE logging: a version whose checkpoint cannot be
    # evaluated must not enter the registry.
    measured = evaluate_checkpoint(
        checkpoint_path, label_file,
        config_path=config_path, dict_path=dict_path,
        max_samples=max_eval_samples,
        legacy_batch_axis_attention=legacy_batch_axis_attention,
    )

    # The semantics the metrics were measured under travel WITH the model:
    # the pyfunc reads this artifact at load time, so a legacy checkpoint
    # can never silently serve under the fixed forward (or vice versa).
    semantics_path = Path(tempfile.mkdtemp(prefix="vin_semantics_")) / "semantics.json"
    semantics_path.write_text(json.dumps({
        "legacy_batch_axis_attention": legacy_batch_axis_attention,
        "forward_contract": (
            "pre-2026-08-20 batch-axis attention (defect reproduction)"
            if legacy_batch_axis_attention else
            "batch-first [B, T, C]; batch-independent (fixed 2026-08-20)"
        ),
    }, indent=1))

    info_path = Path(checkpoint_path).with_name(
        Path(checkpoint_path).stem + "_info.json"
    )

    with mlflow.start_run(run_name=f"register-{stage_label or Path(checkpoint_path).stem}"):
        labels_df = pd.read_csv(
            label_file, sep="\t", names=["path", "vin"], dtype=str
        )
        eval_dataset = mlflow.data.from_pandas(
            labels_df, name=Path(label_file).stem, targets="vin"
        )
        mlflow.log_input(eval_dataset, context="evaluation")

        model_info = mlflow.pyfunc.log_model(
            name=(f"vin-lcnetv3-svtr-ctc-{stage_label}"
                  if stage_label else "vin-lcnetv3-svtr-ctc"),
            python_model=VINRecognizerPyfunc(),
            artifacts={
                "checkpoint": checkpoint_path,
                "config": config_path,
                "char_dict": dict_path,
                "semantics": str(semantics_path),
            },
            params={
                "architecture": "PP-OCRv4 (PPLCNetV3 + SVTR + CTC)",
                "checkpoint": str(checkpoint_path),
                "stage": stage_label,
                "decode": "core.charset.ctc_greedy_decode (blank=0)",
                "legacy_batch_axis_attention": str(legacy_batch_axis_attention),
            },
            registered_model_name=registered_name,
        )

        # Metrics linked to the LoggedModel AND the dataset (MLflow 3
        # LoggedModel workflow).
        mlflow.log_metrics(
            {k: v for k, v in measured.items()},
            model_id=model_info.model_id,
            dataset=eval_dataset,
        )

        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{registered_name}'")
        version = max(int(v.version) for v in versions)
        tags = {
            "checkpoint_path": str(checkpoint_path),
            "stage": stage_label,
            "eval_label_file": label_file,
            "eval_n": str(int(measured["n"])),
            "semantics": (
                "legacy-batch-axis-attention"
                if legacy_batch_axis_attention else "batch-first-fixed"
            ),
        }
        if source_run_id:
            tags["source_training_run"] = source_run_id
        if info_path.is_file():
            tags["checkpoint_info"] = info_path.read_text()[:500]
        for key, value in tags.items():
            client.set_model_version_tag(registered_name, version, key, value)

    return {
        "model_id": model_info.model_id,
        "model_uri": model_info.model_uri,
        "registered_name": registered_name,
        "version": version,
        "metrics": measured,
    }


# =============================================================================
# TRACES - per-stage spans for a single recognition
# =============================================================================

def traced_recognize(
    image_path: str,
    checkpoint_path: str,
    config_path: str = "configs/vin_finetune_config.yml",
    dict_path: str = "configs/vin_dict.txt",
    experiment: str = "vin_finetune",
    legacy_batch_axis_attention: bool = False,
) -> Dict[str, Any]:
    """
    Recognize one image with an MLflow trace: preprocess -> forward ->
    decode -> validate, each a span carrying inputs/outputs and timing.
    View under the experiment's Traces tab.
    """
    import cv2
    import paddle

    from src.vin_ocr.core.charset import ctc_greedy_decode, load_char_dict
    from src.vin_ocr.core.vin_utils import validate_vin
    from src.vin_ocr.training.finetune_paddleocr import VINRecognitionDataset

    mlflow.set_experiment(experiment)
    model, idx_to_char, _ = _load_recognizer(
        checkpoint_path, config_path, dict_path,
        legacy_batch_axis_attention=legacy_batch_axis_attention,
    )
    char_to_idx, _ = load_char_dict(dict_path)

    with mlflow.start_span(name="vin_recognition") as root:
        root.set_inputs({"image_path": image_path, "checkpoint": checkpoint_path})

        with mlflow.start_span(name="preprocess") as span:
            started = time.time()
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"unreadable image: {image_path}")
            dataset = VINRecognitionDataset.__new__(VINRecognitionDataset)
            dataset.img_height, dataset.img_width = 48, 320
            from src.vin_ocr.preprocessing import (
                PreprocessConfig,
                PreprocessStrategy,
                VINPreprocessor,
            )
            dataset._vin_preprocessor = VINPreprocessor(config=PreprocessConfig(
                strategy=PreprocessStrategy.ENGRAVED,
                target_width=960, min_height=48, max_height=192,
            ))
            tensor, valid_width = dataset._preprocess_image(image)
            span.set_inputs({"shape_in": list(image.shape)})
            span.set_outputs({
                "shape_out": list(tensor.shape),
                "valid_width_px": int(valid_width),
                "ms": round((time.time() - started) * 1000, 1),
            })

        with mlflow.start_span(name="forward") as span:
            started = time.time()
            with paddle.no_grad():
                logits = model(paddle.to_tensor(tensor[None]))
            span.set_outputs({
                "logits_shape": list(logits.shape),
                "ms": round((time.time() - started) * 1000, 1),
            })

        with mlflow.start_span(name="decode") as span:
            started = time.time()
            probs = paddle.nn.functional.softmax(logits, axis=-1).numpy()[0]
            text, kept = ctc_greedy_decode(probs.argmax(-1), idx_to_char)
            vin = text[:17]
            confidence = float(probs.max(-1)[kept].mean()) if kept else 0.0
            span.set_outputs({
                "raw_decode": text,
                "vin": vin,
                "confidence": round(confidence, 4),
                "emissions": len(kept),
                "ms": round((time.time() - started) * 1000, 1),
            })

        with mlflow.start_span(name="validate") as span:
            validation = validate_vin(vin)
            span.set_outputs({
                "is_valid_length": validation.is_valid_length,
                "checksum_valid": validation.checksum_valid,
            })

        result = {
            "vin": vin,
            "confidence": confidence,
            "checksum_valid": validation.checksum_valid,
            "raw_decode": text,
        }
        root.set_outputs(result)
    return result
