#!/usr/bin/env python3
"""
Tests for VIN OCR Inference Module
===================================

Tests for both ONNX and Paddle inference backends.
"""

import importlib.util

import pytest
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# paddle and onnxruntime are optional heavy backends. Tests that exercise real
# loader behaviour must skip when they are absent rather than fail with
# ModuleNotFoundError.
requires_paddle = pytest.mark.skipif(
    importlib.util.find_spec("paddle") is None,
    reason="paddlepaddle not installed (optional backend)",
)
requires_onnxruntime = pytest.mark.skipif(
    importlib.util.find_spec("onnxruntime") is None,
    reason="onnxruntime not installed (optional backend)",
)


class TestVINInference:
    """Tests for direct Paddle inference."""
    
    def test_vin_inference_import(self):
        """Test that VINInference can be imported."""
        from src.vin_ocr.inference import VINInference
        assert VINInference is not None
    
    @requires_paddle
    def test_vin_inference_init_missing_model(self):
        """Test that VINInference raises error for missing model."""
        from src.vin_ocr.inference.paddle_inference import VINInference
        
        with pytest.raises(FileNotFoundError):
            VINInference("/nonexistent/model/path")
    
    def test_vin_charset(self):
        """Test VIN charset is correct (no I, O, Q)."""
        from src.vin_ocr.inference.paddle_inference import VIN_CHARSET
        
        assert 'I' not in VIN_CHARSET
        assert 'O' not in VIN_CHARSET
        assert 'Q' not in VIN_CHARSET
        assert len(VIN_CHARSET) == 33  # 10 digits + 23 letters
    
    def test_preprocess_returns_correct_shape(self):
        """Test that preprocess returns correct tensor shape."""
        from src.vin_ocr.inference.paddle_inference import VINInference
        
        # Create a mock model dir with required files
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create dummy files
            Path(tmpdir, "inference.json").touch()
            Path(tmpdir, "inference.pdiparams").touch()
            
            # Mock the _load_model method
            with patch.object(VINInference, '_load_model', return_value=Mock()):
                inference = VINInference(tmpdir)
                
                # Create dummy image
                dummy_img = np.random.randint(0, 255, (100, 400, 3), dtype=np.uint8)
                
                result = inference.preprocess(dummy_img)
                
                assert result.shape == (1, 3, 48, 320)
                assert result.dtype == np.float32
    
    def test_softmax(self):
        """Test softmax computation."""
        from src.vin_ocr.inference.paddle_inference import VINInference
        
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "inference.json").touch()
            Path(tmpdir, "inference.pdiparams").touch()
            
            with patch.object(VINInference, '_load_model', return_value=Mock()):
                inference = VINInference(tmpdir)
                
                x = np.array([[1.0, 2.0, 3.0]])
                result = inference._softmax(x)
                
                assert result.shape == x.shape
                assert np.isclose(result.sum(), 1.0)
                assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_char_dict_loading(self):
        """Test character dictionary loading."""
        from src.vin_ocr.inference.paddle_inference import VINInference, VIN_CHARSET
        
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "inference.json").touch()
            Path(tmpdir, "inference.pdiparams").touch()
            
            with patch.object(VINInference, '_load_model', return_value=Mock()):
                inference = VINInference(tmpdir)
                
                # Default charset should be loaded
                assert '<blank>' in inference.char_dict
                assert inference.char_dict['<blank>'] == 0
                
                # All VIN chars should be present
                for char in VIN_CHARSET:
                    assert char in inference.char_dict


class TestONNXInference:
    """Tests for ONNX inference."""
    
    def test_onnx_inference_import(self):
        """Test that ONNXVINRecognizer can be imported."""
        from src.vin_ocr.inference import ONNXVINRecognizer
        assert ONNXVINRecognizer is not None
    
    def test_onnx_config_defaults(self):
        """Test ONNX config has correct defaults."""
        from src.vin_ocr.inference import ONNXInferenceConfig
        
        config = ONNXInferenceConfig()
        
        assert config.input_height == 48
        assert config.input_width == 320
        assert config.input_channels == 3
        assert config.blank_idx == 0
    
    def test_onnx_vin_charset(self):
        """Test ONNX VIN charset matches."""
        from src.vin_ocr.inference import VIN_CHARSET
        from src.vin_ocr.inference.paddle_inference import VIN_CHARSET as PADDLE_CHARSET
        
        assert VIN_CHARSET == PADDLE_CHARSET
    
    @requires_onnxruntime
    def test_load_onnx_model_missing_file(self):
        """Test that load_onnx_model raises error for missing file."""
        from src.vin_ocr.inference import load_onnx_model
        
        with pytest.raises(FileNotFoundError):
            load_onnx_model("/nonexistent/model.onnx")
    
    @pytest.mark.skipif(
        not Path("output/onnx").exists(),
        reason="ONNX models not available"
    )
    def test_onnx_model_loading(self):
        """Test loading actual ONNX model if available."""
        from src.vin_ocr.inference import ONNXVINRecognizer
        
        onnx_files = list(Path("output/onnx").glob("*.onnx"))
        if onnx_files:
            recognizer = ONNXVINRecognizer(str(onnx_files[0]))
            assert recognizer.session is not None


class TestInferenceIntegration:
    """Integration tests for inference pipeline."""
    
    @pytest.mark.skipif(
        not Path("output").exists(),
        reason="Output directory not available"
    )
    def test_paddle_inference_end_to_end(self):
        """Test end-to-end Paddle inference if model available."""
        from src.vin_ocr.inference import VINInference
        
        # Find an inference model. A servable export needs the GRAPH file
        # (inference.json or inference.pdmodel), not just weights: the
        # weights-only fallback writes inference.pdiparams alone and cannot
        # be loaded by VINInference.
        inference_dirs = list(Path("output").glob("*/inference"))
        valid_dirs = [
            d for d in inference_dirs
            if (d / "inference.pdiparams").exists()
            and ((d / "inference.json").exists() or (d / "inference.pdmodel").exists())
        ]
        
        if not valid_dirs:
            pytest.skip("No servable inference export found (graph file required)")
        
        inference = VINInference(str(valid_dirs[0]))
        
        # Find a test image
        test_images = list(Path("dataset/test").glob("*.jpg"))
        if not test_images:
            test_images = list(Path("dataset/train").glob("*.jpg"))
        
        if not test_images:
            pytest.skip("No test images found")
        
        result = inference.recognize(str(test_images[0]))
        
        assert 'vin' in result
        assert 'confidence' in result
        assert 'raw_text' in result
        assert 'error' in result
        assert result['error'] is None
    
    @pytest.mark.skipif(
        not Path("output/onnx").exists(),
        reason="ONNX models not available"
    )
    def test_onnx_inference_end_to_end(self):
        """Test end-to-end ONNX inference if model available."""
        from src.vin_ocr.inference import ONNXVINRecognizer
        
        onnx_files = list(Path("output/onnx").glob("*.onnx"))
        if not onnx_files:
            pytest.skip("No ONNX models found")
        
        recognizer = ONNXVINRecognizer(str(onnx_files[0]))
        
        # Find a test image
        test_images = list(Path("dataset/test").glob("*.jpg"))
        if not test_images:
            test_images = list(Path("dataset/train").glob("*.jpg"))
        
        if not test_images:
            pytest.skip("No test images found")
        
        result = recognizer.recognize(str(test_images[0]))
        
        assert 'vin' in result
        assert 'confidence' in result
        assert 'raw_text' in result


class TestBatchInference:
    """Tests for batch inference capabilities."""
    
    def test_batch_inference_method_exists(self):
        """Test that batch inference methods exist."""
        from src.vin_ocr.inference import VINInference, ONNXVINRecognizer
        
        assert hasattr(VINInference, 'recognize_batch')
        assert hasattr(ONNXVINRecognizer, 'recognize_batch')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])


class TestErrorContractOrdering:
    """
    A wrong model path must surface as FileNotFoundError - never as an
    ImportError about a runtime the caller may already have. Proven with
    the runtime import BLOCKED, so the ordering is what's under test.
    """

    def _block(self, monkeypatch, module_prefix):
        import builtins
        real_import = builtins.__import__

        def blocked(name, *args, **kwargs):
            if name == module_prefix or name.startswith(module_prefix + "."):
                raise ImportError(f"{module_prefix} not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", blocked)

    def test_onnx_missing_file_is_filenotfound_even_without_runtime(
        self, monkeypatch, tmp_path
    ):
        from src.vin_ocr.inference.onnx_inference import ONNXVINRecognizer

        self._block(monkeypatch, "onnxruntime")
        with pytest.raises(FileNotFoundError):
            ONNXVINRecognizer(str(tmp_path / "no_such_model.onnx"))

    def test_paddle_missing_files_are_filenotfound_even_without_paddle(
        self, monkeypatch, tmp_path
    ):
        from src.vin_ocr.inference.paddle_inference import VINInference

        (tmp_path / "vin_dict.txt").write_text(
            "<blank>\n" + "\n".join("0123456789ABCDEFGHJKLMNPRSTUVWXYZ") + "\n"
        )
        self._block(monkeypatch, "paddle")
        with pytest.raises(FileNotFoundError) as excinfo:
            VINInference(str(tmp_path))
        assert "No model file found" in str(excinfo.value)


class TestNoSourcePathSkipGuards:
    """
    Suite hygiene: a skipif condition referencing a SOURCE-FILE path rots
    when files move - test_pipeline_initialization was silently skipped
    forever because its guard checked the pre-restructure root-level
    vin_pipeline.py. Skip conditions must be capability-based (imports,
    ConfigurationError), never file-layout-based.
    """

    def test_no_skipif_condition_references_a_py_path(self):
        import ast

        offenders = []
        for test_file in Path(__file__).parent.glob("test_*.py"):
            tree = ast.parse(test_file.read_text(encoding="utf-8"))
            source = test_file.read_text(encoding="utf-8")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
                    continue
                for deco in node.decorator_list:
                    if not (
                        isinstance(deco, ast.Call)
                        and isinstance(deco.func, ast.Attribute)
                        and deco.func.attr == "skipif"
                    ):
                        continue
                    segment = ast.get_source_segment(source, deco) or ""
                    for const in ast.walk(deco):
                        if (
                            isinstance(const, ast.Constant)
                            and isinstance(const.value, str)
                            and const.value.endswith(".py")
                        ):
                            offenders.append(
                                f"{test_file.name}:{deco.lineno} -> {segment[:80]}"
                            )
        assert offenders == [], (
            f"source-path-based skip guards rot on restructure: {offenders}"
        )
