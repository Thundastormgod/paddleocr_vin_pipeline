#!/usr/bin/env python3
"""
Streaming-Enabled VIN OCR Training

This script extends the original training to support DagsHub data streaming,
allowing training without local data downloads.

Streaming is implemented with ``dagshub.streaming.install_hooks``: Python's
file-open machinery is patched so that files under the project root which are
not present locally are fetched on demand from the DagsHub repository.
"""

import sys
import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# This script lives AT the repository root.
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from dagshub.streaming import install_hooks
    DAGSHUB_AVAILABLE = True
except ImportError:
    install_hooks = None
    DAGSHUB_AVAILABLE = False

from src.vin_ocr.training.finetune_paddleocr import (
    VINFineTuner,
    load_config,
    merge_config,
)


class DagsHubDataStreamer:
    """Minimal DagsHub streaming setup for this repository.

    Wraps ``dagshub.streaming.install_hooks`` so that data files referenced by
    the training config are streamed from the DagsHub repo when they are not
    present locally.
    """

    def __init__(self, repo_owner: str, repo_name: str):
        self.repo_owner = repo_owner
        self.repo_name = repo_name

    def initialize_streaming(self, username: str, token: str) -> bool:
        """Install streaming hooks for this repo. Returns True on success."""
        if not DAGSHUB_AVAILABLE:
            print("❌ dagshub package not installed")
            return False
        repo_url = f"https://dagshub.com/{self.repo_owner}/{self.repo_name}"
        try:
            install_hooks(
                project_root=project_root,
                repo_url=repo_url,
                username=username,
                token=token,
            )
            return True
        except Exception as e:
            print(f"❌ install_hooks failed for {repo_url}: {type(e).__name__}: {e}")
            return False


class StreamingVINTrainer:
    """VIN OCR trainer with DagsHub streaming support."""

    def __init__(self, config_path: str, use_streaming: bool = False):
        """
        Initialize streaming VIN trainer.

        Args:
            config_path: Path to training configuration
            use_streaming: Whether to use DagsHub streaming
        """
        self.config_path = str(self._resolve_config_path(config_path))
        self.use_streaming = use_streaming
        self.streamer: Optional[DagsHubDataStreamer] = None

        if use_streaming and not DAGSHUB_AVAILABLE:
            raise ImportError("dagshub package required for streaming. Install with: pip install dagshub")

    @staticmethod
    def _resolve_config_path(config_path: str) -> Path:
        """Resolve a config path against the repo root when relative."""
        p = Path(config_path)
        if p.is_absolute():
            return p
        candidate = project_root / p
        return candidate if candidate.exists() else p

    def setup_streaming(self, repo_owner: str, repo_name: str, username: str, token: str):
        """
        Set up DagsHub streaming.

        Args:
            repo_owner: DagsHub repository owner
            repo_name: DagsHub repository name
            username: DagsHub username
            token: DagsHub access token
        """
        if not self.use_streaming:
            return

        self.streamer = DagsHubDataStreamer(repo_owner, repo_name)

        if self.streamer.initialize_streaming(username, token):
            print("✅ DagsHub streaming initialized")

            # Update config paths for streaming
            self._update_config_for_streaming()
        else:
            raise RuntimeError("Failed to initialize DagsHub streaming")

    def _update_config_for_streaming(self):
        """Write a streaming variant of the config with repo-anchored data paths.

        The dataset paths are re-anchored to the repository root, where the
        streaming hooks serve them. The override is DEEP-merged into the
        nested config (Train.dataset..., Eval.dataset...) so sibling keys
        such as Train.loader survive, and the result is written to a NEW
        ``<name>_streaming.<ext>`` file - the original config is never
        overwritten.
        """
        src = Path(self.config_path)
        with open(src, 'r') as f:
            config = yaml.safe_load(f)

        def anchored(path_str: str) -> str:
            p = Path(path_str)
            return str(p if p.is_absolute() else (project_root / p).resolve())

        override: Dict[str, Any] = {}
        for section in ('Train', 'Eval'):
            dataset = config.get(section, {}).get('dataset', {})
            ds_override: Dict[str, Any] = {}
            if 'data_dir' in dataset:
                ds_override['data_dir'] = anchored(dataset['data_dir'])
            if 'label_file_list' in dataset:
                ds_override['label_file_list'] = [
                    anchored(p) for p in dataset['label_file_list']
                ]
            if ds_override:
                override[section] = {'dataset': ds_override}

        merge_config(config, override)

        # Handles both .yml and .yaml: the suffix is preserved, the stem gains
        # a _streaming marker, so the new path can never equal the original.
        streaming_config_path = src.with_name(f"{src.stem}_streaming{src.suffix}")
        with open(streaming_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)

        self.config_path = str(streaming_config_path)
        print(f"✅ Streaming config saved: {streaming_config_path}")

    def train(
        self,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        resume_from: Optional[str] = None,
        use_cpu: bool = False,
    ) -> VINFineTuner:
        """Start training with streaming support."""
        print("🚀 Starting VIN OCR Training")
        print(f"   Config: {self.config_path}")
        print(f"   Streaming: {'✅' if self.use_streaming else '❌'}")

        if self.use_streaming and self.streamer is not None:
            print(f"   DagsHub: {self.streamer.repo_owner}/{self.streamer.repo_name}")

        # VINFineTuner takes a config DICT: load and validate the YAML first,
        # then construct exactly as finetune_paddleocr.main() does.
        config = load_config(self.config_path)

        if use_cpu:
            config['Global']['use_gpu'] = False

        # Apply training overrides (`is not None`: 0 is absent here anyway,
        # but explicit presence checks never mistake falsy values for unset).
        if epochs is not None:
            config['Global']['epoch_num'] = epochs
        if batch_size is not None:
            config['Train']['loader']['batch_size_per_card'] = batch_size
        if learning_rate is not None:
            config['Optimizer']['lr']['learning_rate'] = learning_rate

        trainer = VINFineTuner(
            config=config,
            output_dir=config['Global']['save_model_dir'],
        )

        trainer.train(resume_from=resume_from)

        return trainer


def main():
    """Main training script with streaming support."""
    parser = argparse.ArgumentParser(
        description='VIN OCR Training with DagsHub Streaming Support'
    )

    # Basic training args
    parser.add_argument(
        '--config', '-c',
        default='configs/vin_finetune_config.yml',
        help='Path to config file'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Number of epochs'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=None,
        help='Learning rate'
    )
    parser.add_argument(
        '--resume',
        default=None,
        help='Resume from checkpoint'
    )
    parser.add_argument(
        '--cpu',
        action='store_true',
        help='Force CPU training (sets Global.use_gpu=false in the config)'
    )

    # Streaming args
    parser.add_argument(
        '--stream',
        action='store_true',
        help='Use DagsHub data streaming'
    )
    parser.add_argument(
        '--repo-owner',
        help='DagsHub repository owner (required for streaming)'
    )
    parser.add_argument(
        '--repo-name',
        help='DagsHub repository name (required for streaming)'
    )
    parser.add_argument(
        '--dagshub-user',
        help='DagsHub username (required for streaming)'
    )
    parser.add_argument(
        '--dagshub-token',
        help='DagsHub access token (required for streaming)'
    )

    args = parser.parse_args()

    # Validate streaming args
    if args.stream:
        if not all([args.repo_owner, args.repo_name, args.dagshub_user, args.dagshub_token]):
            print("❌ Streaming requires: --repo-owner, --repo-name, --dagshub-user, --dagshub-token")
            return 1

    try:
        # Initialize streaming trainer
        trainer = StreamingVINTrainer(
            config_path=args.config,
            use_streaming=args.stream
        )

        # Set up streaming if requested
        if args.stream:
            trainer.setup_streaming(
                repo_owner=args.repo_owner,
                repo_name=args.repo_name,
                username=args.dagshub_user,
                token=args.dagshub_token
            )

        # Start training
        trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=args.resume,
            use_cpu=args.cpu,
        )

        print("🎉 Training completed successfully!")
        return 0

    except Exception as e:
        print(f"❌ Training failed: {type(e).__name__}: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
