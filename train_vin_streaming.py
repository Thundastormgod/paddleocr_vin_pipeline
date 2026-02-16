#!/usr/bin/env python3
"""
Streaming-Enabled VIN OCR Training

This script extends the original training to support DagsHub data streaming,
allowing training without local data downloads.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.vin_ocr.data.setup_dagshub_streaming import DagsHubDataStreamer, DAGSHUB_AVAILABLE
from src.vin_ocr.training.finetune_paddleocr import VINFineTuner


class StreamingVINTrainer:
    """VIN OCR trainer with DagsHub streaming support."""
    
    def __init__(self, config_path: str, use_streaming: bool = False):
        """
        Initialize streaming VIN trainer.
        
        Args:
            config_path: Path to training configuration
            use_streaming: Whether to use DagsHub streaming
        """
        self.config_path = config_path
        self.use_streaming = use_streaming
        self.streamer: Optional[DagsHubDataStreamer] = None
        
        if use_streaming and not DAGSHUB_AVAILABLE:
            raise ImportError("dagshub package required for streaming. Install with: pip install dagshub")
    
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
        """Update configuration paths for streaming data access."""
        import yaml
        
        # Load current config
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update data paths to use streaming
        if self.streamer:
            streaming_paths = {
                'Train': {
                    'dataset': {
                        'data_dir': f"/mnt/dagsHub/{self.streamer.repo_owner}/{self.streamer.repo_name}/finetune_data/train_images",
                        'label_file_list': [f"/mnt/dagsHub/{self.streamer.repo_owner}/{self.streamer.repo_name}/finetune_data/train_labels.txt"]
                    }
                },
                'Eval': {
                    'dataset': {
                        'data_dir': f"/mnt/dagsHub/{self.streamer.repo_owner}/{self.streamer.repo_name}/finetune_data/val_images",
                        'label_file_list': [f"/mnt/dagsHub/{self.streamer.repo_owner}/{self.streamer.repo_name}/finetune_data/val_labels.txt"]
                    }
                }
            }
            
            config.update(streaming_paths)
            
            # Save updated config
            streaming_config_path = self.config_path.replace('.yml', '_streaming.yml')
            with open(streaming_config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            self.config_path = streaming_config_path
            print(f"✅ Streaming config saved: {streaming_config_path}")
    
    def train(self, **kwargs):
        """Start training with streaming support."""
        print(f"🚀 Starting VIN OCR Training")
        print(f"   Config: {self.config_path}")
        print(f"   Streaming: {'✅' if self.use_streaming else '❌'}")
        
        if self.use_streaming:
            print(f"   DagsHub: {self.streamer.repo_owner}/{self.streamer.repo_name}")
        
        # Initialize trainer with updated config
        trainer = VINFineTuner(self.config_path)
        
        # Apply training overrides
        if kwargs.get('epochs'):
            trainer.config['Global']['epoch_num'] = kwargs['epochs']
        if kwargs.get('batch_size'):
            trainer.config['Train']['loader']['batch_size_per_card'] = kwargs['batch_size']
        if kwargs.get('learning_rate'):
            trainer.config['Optimizer']['lr']['learning_rate'] = kwargs['learning_rate']
        
        # Start training
        trainer.train(resume_from=kwargs.get('resume_from'))
        
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
        help='Force CPU training'
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
        trained_model = trainer.train(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            resume_from=args.resume
        )
        
        print("🎉 Training completed successfully!")
        return 0
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return 1


if __name__ == '__main__':
    sys.exit(main())
