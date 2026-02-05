"""
DeepSeek Model ONNX Export Script

This script exports fine-tuned DeepSeek models to ONNX format for
portable, production-ready inference.

⚠️ REQUIREMENTS:
- Fine-tuned DeepSeek model checkpoint
- HPC with NVIDIA RTX 3090 (24GB VRAM) or similar
- torch, transformers, onnx packages

Usage (on HPC with RTX 3090):
    python -m src.vin_ocr.training.export_deepseek_onnx \\
        --model-path output/deepseek_finetune/best_model \\
        --output-dir models/deepseek_onnx

After export, copy ONNX files to local machine for inference:
    scp -r user@hpc:models/deepseek_onnx/ ./models/deepseek_onnx/

Note: RTX 3090 (24GB) can handle DeepSeek-VL 7B with FP16.
      For larger models, use --load-in-8bit during fine-tuning.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

# Project root for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def check_dependencies():
    """Check if required dependencies are available."""
    missing = []
    
    try:
        import torch
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
    except ImportError:
        missing.append("transformers")
    
    try:
        import onnx
    except ImportError:
        missing.append("onnx")
    
    if missing:
        print("=" * 60)
        print("ERROR: Missing required dependencies")
        print("=" * 60)
        print(f"\nMissing packages: {', '.join(missing)}")
        print("\nInstall with:")
        print(f"  pip install {' '.join(missing)}")
        print("\n⚠️ This script is designed to run on HPC with CUDA.")
        print("=" * 60)
        return False
    
    return True


class DeepSeekONNXExporter:
    """Export fine-tuned DeepSeek models to ONNX format."""
    
    def __init__(self, model_path: str, output_dir: str):
        """
        Initialize the exporter.
        
        Args:
            model_path: Path to fine-tuned DeepSeek model directory
            output_dir: Directory for ONNX output files
        """
        self.model_path = Path(model_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate model path
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path not found: {model_path}")
        
        # Check for model files
        has_pytorch = (self.model_path / "pytorch_model.bin").exists()
        has_safetensors = (self.model_path / "model.safetensors").exists()
        
        if not has_pytorch and not has_safetensors:
            raise FileNotFoundError(
                f"No model checkpoint found in {model_path}. "
                "Expected pytorch_model.bin or model.safetensors"
            )
        
        print(f"Model path: {self.model_path}")
        print(f"Output dir: {self.output_dir}")
    
    def load_model(self):
        """Load the fine-tuned DeepSeek model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, AutoProcessor
        
        print("\nLoading fine-tuned DeepSeek model...")
        
        # Try to load with different configurations
        try:
            # First try loading as a vision-language model
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16,
                trust_remote_code=True,
                device_map="auto"
            )
            print("  ✓ Model loaded")
        except Exception as e:
            print(f"  ✗ Failed to load model: {e}")
            raise
        
        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            print("  ✓ Tokenizer loaded")
        except Exception:
            # Try loading from base model
            config_path = self.model_path / "config.json"
            if config_path.exists():
                with open(config_path) as f:
                    config = json.load(f)
                base_model = config.get("_name_or_path", "deepseek-ai/deepseek-vl-7b-base")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    base_model,
                    trust_remote_code=True
                )
                print(f"  ✓ Tokenizer loaded from base: {base_model}")
        
        # Try to load processor for image handling
        try:
            self.processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True
            )
            print("  ✓ Processor loaded")
        except Exception:
            self.processor = None
            print("  ⚠ No processor found (will use default preprocessing)")
        
        # Put model in eval mode
        self.model.eval()
        
        return True
    
    def export_to_onnx(
        self,
        input_height: int = 384,
        input_width: int = 384,
        opset_version: int = 14
    ):
        """
        Export the model to ONNX format.
        
        Args:
            input_height: Input image height
            input_width: Input image width
            opset_version: ONNX opset version
        """
        import torch
        import onnx
        
        print(f"\nExporting to ONNX (opset={opset_version})...")
        
        # Create dummy inputs
        batch_size = 1
        
        # For vision models, we need image tensor
        dummy_image = torch.randn(
            batch_size, 3, input_height, input_width,
            dtype=torch.float16
        ).to(self.model.device)
        
        # Some models also need text tokens
        dummy_input_ids = torch.zeros(
            batch_size, 1,
            dtype=torch.long
        ).to(self.model.device)
        
        # Output path
        onnx_path = self.output_dir / "deepseek_finetuned.onnx"
        
        try:
            # Try exporting with image input only
            torch.onnx.export(
                self.model,
                (dummy_image,),
                str(onnx_path),
                input_names=['pixel_values'],
                output_names=['logits'],
                dynamic_axes={
                    'pixel_values': {0: 'batch_size'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=opset_version,
                do_constant_folding=True,
            )
        except Exception as e:
            print(f"  ⚠ Simple export failed: {e}")
            print("  Trying with combined inputs...")
            
            # Try with both image and text inputs
            try:
                torch.onnx.export(
                    self.model,
                    (dummy_input_ids, dummy_image),
                    str(onnx_path),
                    input_names=['input_ids', 'pixel_values'],
                    output_names=['logits'],
                    dynamic_axes={
                        'input_ids': {0: 'batch_size', 1: 'sequence'},
                        'pixel_values': {0: 'batch_size'},
                        'logits': {0: 'batch_size'}
                    },
                    opset_version=opset_version,
                    do_constant_folding=True,
                )
            except Exception as e2:
                print(f"  ✗ ONNX export failed: {e2}")
                print("\n  Note: Vision-Language models can be complex to export.")
                print("  Consider using optimum library for better compatibility:")
                print("    pip install optimum[exporters]")
                print("    optimum-cli export onnx --model <model_path> --task causal-lm-with-past <output>")
                raise
        
        print(f"  ✓ ONNX model saved: {onnx_path}")
        
        # Validate the exported model
        self._validate_onnx(onnx_path)
        
        # Save metadata
        self._save_metadata(input_height, input_width, opset_version)
        
        return onnx_path
    
    def _validate_onnx(self, onnx_path: Path):
        """Validate the exported ONNX model."""
        import onnx
        
        print("\nValidating ONNX model...")
        
        try:
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("  ✓ ONNX model validation passed")
            
            # Print model info
            print(f"\n  Model info:")
            print(f"    IR version: {onnx_model.ir_version}")
            print(f"    Opset: {onnx_model.opset_import[0].version}")
            print(f"    Inputs: {[i.name for i in onnx_model.graph.input]}")
            print(f"    Outputs: {[o.name for o in onnx_model.graph.output]}")
            
        except Exception as e:
            print(f"  ⚠ Validation warning: {e}")
    
    def _save_metadata(self, input_height: int, input_width: int, opset_version: int):
        """Save export metadata."""
        metadata = {
            "source_model": str(self.model_path),
            "model_type": "deepseek_finetuned",
            "input_height": input_height,
            "input_width": input_width,
            "opset_version": opset_version,
            "format": "onnx",
            "notes": "Fine-tuned DeepSeek model for VIN recognition"
        }
        
        metadata_path = self.output_dir / "model_info.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"  ✓ Metadata saved: {metadata_path}")
    
    def test_inference(self, test_image_path: Optional[str] = None):
        """Test ONNX inference with a sample image."""
        try:
            import onnxruntime as ort
            import numpy as np
            
            print("\nTesting ONNX inference...")
            
            onnx_path = self.output_dir / "deepseek_finetuned.onnx"
            session = ort.InferenceSession(str(onnx_path))
            
            # Get input info
            input_info = session.get_inputs()[0]
            print(f"  Input: {input_info.name}, shape: {input_info.shape}")
            
            # Create dummy input
            input_shape = input_info.shape
            if isinstance(input_shape[2], int) and isinstance(input_shape[3], int):
                h, w = input_shape[2], input_shape[3]
            else:
                h, w = 384, 384
            
            dummy_input = np.random.randn(1, 3, h, w).astype(np.float32)
            
            # Run inference
            outputs = session.run(None, {input_info.name: dummy_input})
            
            print(f"  ✓ Inference successful")
            print(f"  Output shape: {outputs[0].shape}")
            
        except ImportError:
            print("  ⚠ onnxruntime not available, skipping inference test")
        except Exception as e:
            print(f"  ⚠ Inference test failed: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Export fine-tuned DeepSeek model to ONNX format"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to fine-tuned DeepSeek model directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models/deepseek_onnx",
        help="Output directory for ONNX files"
    )
    parser.add_argument(
        "--input-height",
        type=int,
        default=384,
        help="Input image height (default: 384)"
    )
    parser.add_argument(
        "--input-width",
        type=int,
        default=384,
        help="Input image width (default: 384)"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=14,
        help="ONNX opset version (default: 14)"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run inference test after export"
    )
    
    args = parser.parse_args()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    print("=" * 60)
    print("DeepSeek ONNX Export")
    print("=" * 60)
    
    try:
        exporter = DeepSeekONNXExporter(args.model_path, args.output_dir)
        exporter.load_model()
        onnx_path = exporter.export_to_onnx(
            input_height=args.input_height,
            input_width=args.input_width,
            opset_version=args.opset
        )
        
        if args.test:
            exporter.test_inference()
        
        print("\n" + "=" * 60)
        print("✓ Export complete!")
        print("=" * 60)
        print(f"\nONNX model: {onnx_path}")
        print("\nTo use on local machine:")
        print(f"  scp -r user@hpc:{args.output_dir}/ ./models/deepseek_onnx/")
        print("\nFor inference:")
        print("  pip install onnxruntime  # or onnxruntime-gpu")
        
    except Exception as e:
        print(f"\n✗ Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
