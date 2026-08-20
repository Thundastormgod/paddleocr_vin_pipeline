"""
Rosetta-ResNet34 for PyTorch/MPS: the Apple-GPU training architecture.

Why this exists (decision record, 2026-08-20): training on this Apple
machine is a long-term requirement. PaddlePaddle has no Metal backend and
never will (its GPU support is CUDA/ROCm), so the TRAINING stack moves to
PyTorch + MPS. Paddle remains as the frozen evaluator for archived
checkpoints and as the inference engine (PaddleOCR is the production OCR).
Measured on this machine before this module was written: paddle CPU is
pinned at ~1.1 of 8 cores regardless of OMP settings (4.82/4.40/4.41
s/step at 1/4/8 threads); torch MPS conv+backward verified working;
``aten::_ctc_loss`` verified NOT implemented on MPS - hence the CPU-bridge
in the trainer.

Architecture - Rosetta (Borisyuk, Gordo & Sivakumar, KDD 2018): a
convolutional backbone whose column features feed CTC directly; NO
sequence module, so every timestep is predicted from its receptive field
alone and the model is batch-independent by construction (in eval mode;
BatchNorm batch statistics apply during training, as in any BN network).

Backbone: torchvision ResNet-34 with stride surgery instead of this
repo's paddle ResNet34-vd. Deliberate: torchvision carries
ImageNet-pretrained weights, and pretrained warm starts are this repo's #1
measured lever (from-scratch was refuted 82.70% vs 66.61% val char
accuracy). Stride changes do not invalidate pretrained kernels. The two
Rosetta variants therefore differ in stem (7x7 vs deep-vd) and are named
distinctly wherever they are compared.

Geometry (48x320 input): stem s2 -> 24x160, maxpool s2 -> 12x80,
layer1 -> 12x80, layer2 (2,1) -> 6x80, layer3 (2,1) -> 3x80,
layer4 (2,1) -> 2x80; height-pool -> T=80 columns = width/4, above the
2*17+1 = 35 CTC feasibility bound this repo once violated.
"""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn

#: Timestep count produced from a 320px-wide input; single source of truth
#: for the trainer's per-sample CTC input lengths.
OUTPUT_TIMESTEPS = 80
INPUT_HEIGHT = 48
INPUT_WIDTH = 320
MIN_CTC_TIMESTEPS = 2 * 17 + 1


def _apply_stride_surgery(block: nn.Module, stride: Tuple[int, int]) -> None:
    """
    Change a torchvision BasicBlock's downsampling stride in place.

    torchvision BasicBlock carries its stride on conv1 and on
    downsample[0] (the 1x1 projection). Changing stride preserves the
    pretrained kernels; only the sampling grid changes.
    """
    assert hasattr(block, "conv1") and block.downsample is not None, (
        "stride surgery targets the first (projection) block of a stage"
    )
    block.conv1.stride = stride
    block.downsample[0].stride = stride
    block.stride = stride


class RosettaResNet34Torch(nn.Module):
    """
    torchvision ResNet-34 backbone (optional ImageNet-1k weights) ->
    height average pool -> dropout -> per-column Linear -> CTC logits.

    Args:
        num_classes: CTC classes including blank at index 0 (repo charset
            contract: blank=0).
        pretrained_backbone: Load torchvision IMAGENET1K_V1 weights
            (downloads ~83MB to ~/.cache/torch on first use). False gives
            a scratch-initialized twin for honest scratch-vs-warmstart
            comparisons.
        dropout: Head dropout probability.
    """

    def __init__(self, num_classes: int, pretrained_backbone: bool = True,
                 dropout: float = 0.1):
        super().__init__()
        assert num_classes >= 2, f"CTC needs blank + alphabet, got {num_classes}"
        self.num_classes = num_classes
        self.pretrained_backbone = pretrained_backbone

        from torchvision.models import ResNet34_Weights, resnet34
        weights = ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None
        net = resnet34(weights=weights)

        # Height-only downsampling in the three strided stages: W/4 is
        # preserved end-to-end, H collapses toward the pool.
        for stage in (net.layer2, net.layer3, net.layer4):
            _apply_stride_surgery(stage[0], (2, 1))

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        # net.avgpool / net.fc are the ImageNet classification head - unused.

        self.pool = nn.AdaptiveAvgPool2d((1, None))
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 4, f"expected [B, C, H, W], got rank {x.ndim}"
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)                 # [B, 512, H', W/4]
        x = self.pool(x)                   # [B, 512, 1, T]
        x = x.squeeze(2).permute(0, 2, 1)  # [B, T, 512]
        assert x.shape[1] >= MIN_CTC_TIMESTEPS, (
            f"T={x.shape[1]} timesteps cannot carry a 17-char CTC target "
            f"(needs >= {MIN_CTC_TIMESTEPS}); input width too small?"
        )
        return self.head(self.dropout(x))  # [B, T, num_classes]
