#!/usr/bin/env python3
"""
Re-export PaddlePaddle Models and Convert to ONNX
==================================================

This script:
1. Loads the saved .pdiparams weights
2. Recreates the model architecture
3. Exports to static graph (.pdmodel + .pdiparams)
4. Converts to ONNX

This fixes the issue where models only have .pdiparams without .pdmodel
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import paddle
import paddle.nn as nn
import numpy as np


def create_vin_model():
    """Create VIN recognition model architecture."""
    
    class HardSwish(nn.Layer):
        def forward(self, x):
            return x * nn.functional.relu6(x + 3) / 6
    
    class SEBlock(nn.Layer):
        def __init__(self, channels, reduction=4):
            super().__init__()
            mid_channels = channels // reduction
            self.pool = nn.AdaptiveAvgPool2D(1)
            self.fc1 = nn.Conv2D(channels, mid_channels, 1)
            self.fc2 = nn.Conv2D(mid_channels, channels, 1)
        
        def forward(self, x):
            identity = x
            x = self.pool(x)
            x = nn.functional.relu(self.fc1(x))
            x = nn.functional.hardsigmoid(self.fc2(x))
            return identity * x
    
    class DepthwiseSeparableConv(nn.Layer):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_se=False):
            super().__init__()
            padding = kernel_size // 2
            self.depthwise = nn.Conv2D(in_channels, in_channels, kernel_size,
                                       stride=stride, padding=padding, groups=in_channels)
            self.bn1 = nn.BatchNorm2D(in_channels)
            self.pointwise = nn.Conv2D(in_channels, out_channels, 1)
            self.bn2 = nn.BatchNorm2D(out_channels)
            self.act = HardSwish()
            self.use_se = use_se
            if use_se:
                self.se = SEBlock(out_channels)
        
        def forward(self, x):
            x = self.act(self.bn1(self.depthwise(x)))
            x = self.bn2(self.pointwise(x))
            if self.use_se:
                x = self.se(x)
            return self.act(x)
    
    class PPLCNetV3Backbone(nn.Layer):
        NET_CONFIG = [
            [3, 16, 32, 1, False],
            [3, 32, 64, 2, False],
            [3, 64, 64, 1, False],
            [3, 64, 128, (2, 1), False],
            [3, 128, 128, 1, True],
            [3, 128, 256, (2, 1), False],
            [5, 256, 256, 1, True],
            [5, 256, 256, 1, True],
            [3, 256, 512, (2, 1), True],
            [5, 512, 512, 1, True],
            [5, 512, 512, 1, True],
        ]
        
        def __init__(self, in_channels=3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2D(in_channels, 16, 3, stride=2, padding=1),
                nn.BatchNorm2D(16),
                HardSwish()
            )
            layers = []
            for k, in_c, out_c, s, se in self.NET_CONFIG:
                layers.append(DepthwiseSeparableConv(in_c, out_c, k, s, se))
            self.stages = nn.Sequential(*layers)
            self.out_channels = 512
        
        def forward(self, x):
            x = self.stem(x)
            x = self.stages(x)
            return x
    
    class SVTREncoder(nn.Layer):
        def __init__(self, in_channels=512, hidden_dim=256, num_heads=8, num_layers=2, dropout=0.1):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2D((1, None))
            self.proj = nn.Linear(in_channels, hidden_dim)
            self.pos_embed = nn.Embedding(200, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads,
                dim_feedforward=hidden_dim * 4, dropout=dropout, activation='gelu'
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.out_channels = hidden_dim
        
        def forward(self, x):
            x = self.pool(x)
            x = x.squeeze(2).transpose([0, 2, 1])
            x = self.proj(x)
            T = x.shape[1]
            positions = paddle.arange(T).unsqueeze(0).expand([x.shape[0], -1])
            x = x + self.pos_embed(positions)
            x = x.transpose([1, 0, 2])
            x = self.transformer(x)
            x = x.transpose([1, 0, 2])
            return x
    
    class CTCHead(nn.Layer):
        def __init__(self, in_channels, num_classes, dropout=0.1):
            super().__init__()
            self.fc1 = nn.Linear(in_channels, in_channels)
            self.dropout = nn.Dropout(dropout)
            self.fc2 = nn.Linear(in_channels, num_classes)
        
        def forward(self, x):
            x = nn.functional.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
    
    class VINRecognitionModel(nn.Layer):
        def __init__(self, num_classes=34):
            super().__init__()
            self.backbone = PPLCNetV3Backbone(in_channels=3)
            self.neck = SVTREncoder(in_channels=512, hidden_dim=256, num_heads=8, num_layers=2)
            self.head = CTCHead(in_channels=256, num_classes=num_classes)
        
        def forward(self, x):
            features = self.backbone(x)
            sequence = self.neck(features)
            logits = self.head(sequence)
            return logits
    
    return VINRecognitionModel(num_classes=34)


def export_model_to_onnx(pdiparams_path: str, output_dir: str):
    """Export a single model to ONNX."""
    import paddle2onnx
    import onnx
    
    pdiparams_path = Path(pdiparams_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = pdiparams_path.parent.name
    
    print(f"\n📁 Processing: {model_name}")
    
    # Create model
    print("   Creating model architecture...")
    model = create_vin_model()
    
    # Load weights from .pdparams (checkpoint file)
    print(f"   Loading weights from: {pdiparams_path.name}")
    try:
        state_dict = paddle.load(str(pdiparams_path), return_numpy=True)
        # Convert numpy arrays to tensors
        state_dict_tensors = {k: paddle.to_tensor(v) for k, v in state_dict.items()}
        model.set_state_dict(state_dict_tensors)
        print(f"   ✅ Loaded {len(state_dict)} parameters")
    except Exception as e:
        print(f"   ❌ Failed to load weights: {e}")
        return None
    
    model.eval()
    
    # Export to static graph
    temp_dir = output_dir / "temp"
    temp_dir.mkdir(exist_ok=True)
    temp_model_path = temp_dir / "inference"
    
    print("   Exporting to static graph...")
    input_spec = [
        paddle.static.InputSpec(shape=[1, 3, 48, 320], dtype='float32', name='x')
    ]
    
    try:
        paddle.jit.save(model, str(temp_model_path), input_spec=input_spec)
        print(f"   ✅ Static graph exported")
    except Exception as e:
        print(f"   ❌ Static graph export failed: {e}")
        return None
    
    # Convert to ONNX
    # Check which format was exported (.pdmodel for older paddle, .json for newer)
    pdmodel_path = str(temp_model_path) + ".pdmodel"
    pdjson_path = str(temp_model_path) + ".json"
    pdiparams_temp = str(temp_model_path) + ".pdiparams"
    onnx_path = output_dir / f"{model_name}.onnx"
    
    # Use .json if .pdmodel doesn't exist (newer Paddle versions)
    if os.path.exists(pdjson_path) and not os.path.exists(pdmodel_path):
        pdmodel_path = pdjson_path
    
    print("   Converting to ONNX...")
    print(f"      Model file: {os.path.basename(pdmodel_path)}")
    try:
        paddle2onnx.export(
            pdmodel_path,
            pdiparams_temp,
            str(onnx_path),
            opset_version=11,
            auto_upgrade_opset=True,
            verbose=False,
        )
    except Exception as e:
        print(f"   ❌ ONNX conversion failed: {e}")
        return None
    
    # Verify ONNX
    try:
        onnx_model = onnx.load(str(onnx_path))
        onnx.checker.check_model(onnx_model)
        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        print(f"   ✅ ONNX model created: {onnx_path.name} ({size_mb:.1f} MB)")
    except Exception as e:
        print(f"   ⚠️  ONNX verification warning: {e}")
    
    # Cleanup temp files
    for f in temp_dir.glob("*"):
        f.unlink()
    temp_dir.rmdir()
    
    return str(onnx_path)


def test_onnx_model(onnx_path: str):
    """Test ONNX model with dummy input."""
    import onnxruntime as ort
    
    print(f"\n   Testing inference...")
    session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    
    input_info = session.get_inputs()[0]
    print(f"   Input: {input_info.name} {input_info.shape}")
    
    # Create test input
    dummy_input = np.random.randn(1, 3, 48, 320).astype(np.float32)
    
    # Run inference
    outputs = session.run(None, {input_info.name: dummy_input})
    print(f"   Output shape: {outputs[0].shape}")
    print(f"   ✅ Inference successful!")
    
    return True


def convert_all():
    """Convert all models in output directory."""
    output_base = Path("output")
    onnx_dir = Path("output/onnx")
    onnx_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Converting All Models to ONNX")
    print("=" * 60)
    
    # Find all latest.pdparams files (actual model checkpoints)
    pdiparams_files = list(output_base.glob("*/latest.pdparams"))
    print(f"Found {len(pdiparams_files)} models to convert\n")
    
    converted = []
    failed = []
    
    for pdiparams_path in pdiparams_files:
        try:
            onnx_path = export_model_to_onnx(pdiparams_path, onnx_dir)
            if onnx_path:
                test_onnx_model(onnx_path)
                converted.append(onnx_path)
            else:
                failed.append(str(pdiparams_path))
        except Exception as e:
            print(f"   ❌ Error: {e}")
            failed.append(str(pdiparams_path))
    
    # Summary
    print("\n" + "=" * 60)
    print("Conversion Summary")
    print("=" * 60)
    print(f"  ✅ Converted: {len(converted)}")
    print(f"  ❌ Failed: {len(failed)}")
    
    if converted:
        print(f"\n  ONNX models saved to: {onnx_dir}/")
        for path in converted:
            print(f"    - {Path(path).name}")
    
    return converted, failed


if __name__ == "__main__":
    converted, failed = convert_all()
