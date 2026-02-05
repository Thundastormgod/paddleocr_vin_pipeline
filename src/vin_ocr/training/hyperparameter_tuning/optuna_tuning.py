"""
Optuna Hyperparameter Tuning for VIN OCR Models
================================================

Provides automated hyperparameter optimization using Optuna for:
- PaddleOCR models (PP-OCRv5, CRNN, SVTR_LCNet, SVTR_Tiny)
- DeepSeek Vision-Language models

Features:
- Bayesian optimization with pruning
- Multi-objective optimization (accuracy + speed)
- Distributed training support
- Checkpoint saving and resumption

Author: VIN OCR Pipeline
License: MIT
"""

import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner, HyperbandPruner
import json
import logging
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)


# =============================================================================
# SEARCH SPACE DEFINITIONS
# =============================================================================

@dataclass
class PaddleOCRSearchSpace:
    """
    Search space for PaddleOCR model hyperparameters.
    
    Defines the ranges and distributions for hyperparameter search.
    """
    # Architecture choices
    architectures: List[str] = field(default_factory=lambda: [
        'PP-OCRv5', 'SVTR_LCNet', 'SVTR_Tiny', 'CRNN'
    ])
    
    # Learning rate
    lr_min: float = 1e-5
    lr_max: float = 1e-2
    lr_log: bool = True  # Use log scale
    
    # Batch size
    batch_sizes: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    
    # Epochs
    epochs_min: int = 5
    epochs_max: int = 50
    
    # Optimizer
    optimizers: List[str] = field(default_factory=lambda: ['Adam', 'AdamW', 'SGD'])
    
    # Weight decay
    weight_decay_min: float = 1e-6
    weight_decay_max: float = 1e-2
    
    # Warmup ratio
    warmup_ratio_min: float = 0.0
    warmup_ratio_max: float = 0.2
    
    # Label smoothing
    label_smoothing_min: float = 0.0
    label_smoothing_max: float = 0.2
    
    # Dropout (for SVTR models)
    dropout_min: float = 0.0
    dropout_max: float = 0.5
    
    # Image size
    image_heights: List[int] = field(default_factory=lambda: [32, 48, 64])
    image_widths: List[int] = field(default_factory=lambda: [128, 192, 256, 320])


@dataclass
class DeepSeekSearchSpace:
    """Search space for DeepSeek model hyperparameters."""
    
    # Learning rate
    lr_min: float = 1e-6
    lr_max: float = 1e-4
    lr_log: bool = True
    
    # Batch size (smaller due to model size)
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4])
    
    # Epochs
    epochs_min: int = 1
    epochs_max: int = 10
    
    # LoRA parameters
    lora_r_options: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    lora_alpha_options: List[int] = field(default_factory=lambda: [8, 16, 32, 64])
    lora_dropout_min: float = 0.0
    lora_dropout_max: float = 0.2
    
    # Gradient accumulation
    gradient_accumulation_options: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    
    # Weight decay
    weight_decay_min: float = 0.0
    weight_decay_max: float = 0.1
    
    # Warmup ratio
    warmup_ratio_min: float = 0.0
    warmup_ratio_max: float = 0.1


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning."""
    
    # Study settings
    study_name: str = "vin_ocr_tuning"
    storage: Optional[str] = None  # SQLite URL for persistence
    direction: str = "maximize"  # 'maximize' for accuracy
    
    # Optimization settings
    n_trials: int = 50
    timeout: Optional[int] = None  # Seconds
    n_jobs: int = 1  # Parallel trials
    
    # Sampler settings
    sampler_seed: int = 42
    n_startup_trials: int = 10  # Random trials before TPE
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_warmup_steps: int = 5
    pruning_warmup_trials: int = 5
    
    # Data settings
    train_data_dir: str = "./data/train"
    train_labels: str = "./data/train/labels.txt"
    val_data_dir: str = "./data/val"
    val_labels: str = "./data/val/labels.txt"
    
    # Output settings
    output_dir: str = "./output/hyperparameter_tuning"
    save_best_model: bool = True
    
    # Early stopping
    early_stopping_trials: int = 10  # Stop if no improvement
    
    # Resource limits
    max_time_per_trial: int = 3600  # 1 hour per trial
    
    # Metric to optimize
    metric: str = "val_accuracy"


# =============================================================================
# OBJECTIVE FUNCTIONS
# =============================================================================

class PaddleOCRObjective:
    """Optuna objective function for PaddleOCR models."""
    
    def __init__(
        self,
        search_space: PaddleOCRSearchSpace,
        config: TuningConfig,
        device: str = "cpu",
    ):
        self.search_space = search_space
        self.config = config
        self.device = device
    
    def __call__(self, trial: Trial) -> float:
        """
        Objective function called by Optuna for each trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation accuracy (metric to maximize)
        """
        # Sample hyperparameters
        params = self._sample_params(trial)
        
        logger.info(f"Trial {trial.number}: {params}")
        
        try:
            # Train model with sampled parameters
            accuracy = self._train_and_evaluate(params, trial)
            
            return accuracy
            
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    def _sample_params(self, trial: Trial) -> Dict[str, Any]:
        """Sample hyperparameters from search space."""
        params = {
            'architecture': trial.suggest_categorical(
                'architecture', 
                self.search_space.architectures
            ),
            'learning_rate': trial.suggest_float(
                'learning_rate',
                self.search_space.lr_min,
                self.search_space.lr_max,
                log=self.search_space.lr_log
            ),
            'batch_size': trial.suggest_categorical(
                'batch_size',
                self.search_space.batch_sizes
            ),
            'epochs': trial.suggest_int(
                'epochs',
                self.search_space.epochs_min,
                self.search_space.epochs_max
            ),
            'optimizer': trial.suggest_categorical(
                'optimizer',
                self.search_space.optimizers
            ),
            'weight_decay': trial.suggest_float(
                'weight_decay',
                self.search_space.weight_decay_min,
                self.search_space.weight_decay_max,
                log=True
            ),
            'warmup_ratio': trial.suggest_float(
                'warmup_ratio',
                self.search_space.warmup_ratio_min,
                self.search_space.warmup_ratio_max
            ),
            'label_smoothing': trial.suggest_float(
                'label_smoothing',
                self.search_space.label_smoothing_min,
                self.search_space.label_smoothing_max
            ),
            'image_height': trial.suggest_categorical(
                'image_height',
                self.search_space.image_heights
            ),
            'image_width': trial.suggest_categorical(
                'image_width',
                self.search_space.image_widths
            ),
        }
        
        # Architecture-specific parameters
        if params['architecture'] in ['SVTR_LCNet', 'SVTR_Tiny']:
            params['dropout'] = trial.suggest_float(
                'dropout',
                self.search_space.dropout_min,
                self.search_space.dropout_max
            )
        
        return params
    
    def _train_and_evaluate(
        self, 
        params: Dict[str, Any],
        trial: Trial,
    ) -> float:
        """Train model and return validation accuracy."""
        from src.vin_ocr.training.train_from_scratch import (
            PaddleOCRScratchTrainer,
            PaddleOCRScratchConfig,
        )
        
        # Convert warmup_ratio to warmup_epochs (approximate)
        warmup_epochs = int(params['epochs'] * params.get('warmup_ratio', 0.1))
        
        # Create training config matching PaddleOCRScratchConfig fields
        train_config = PaddleOCRScratchConfig(
            architecture=params['architecture'],
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            num_epochs=params['epochs'],
            weight_decay=params['weight_decay'],
            warmup_epochs=max(1, warmup_epochs),
            label_smoothing=params.get('label_smoothing', 0.1),
            image_height=params['image_height'],
            image_width=params['image_width'],
            dropout=params.get('dropout', 0.1),
            optimizer=params.get('optimizer', 'Adam'),
            # Data paths - matching PaddleOCRScratchConfig field names
            train_data_dir=self.config.train_data_dir,
            train_label_file=self.config.train_labels,
            val_data_dir=self.config.val_data_dir if hasattr(self.config, 'val_data_dir') else self.config.train_data_dir,
            val_label_file=self.config.val_labels,
            # Output
            output_dir=os.path.join(
                self.config.output_dir, 
                f"trial_{trial.number}"
            ),
            # Device
            use_gpu=(self.device != 'cpu'),
        )
        
        # Create trainer with pruning callback
        trainer = PaddleOCRScratchTrainer(train_config)
        
        # Add pruning callback
        def epoch_callback(epoch: int, metrics: Dict[str, float]):
            val_acc = metrics.get('val_accuracy', 0.0)
            trial.report(val_acc, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Train
        try:
            best_accuracy = trainer.train(epoch_callback=epoch_callback)
            return best_accuracy
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 0.0


class DeepSeekObjective:
    """Optuna objective function for DeepSeek models."""
    
    def __init__(
        self,
        search_space: DeepSeekSearchSpace,
        config: TuningConfig,
        device: str = "cpu",
    ):
        self.search_space = search_space
        self.config = config
        self.device = device
    
    def __call__(self, trial: Trial) -> float:
        """Objective function for DeepSeek model tuning."""
        params = self._sample_params(trial)
        
        logger.info(f"Trial {trial.number}: {params}")
        
        try:
            accuracy = self._train_and_evaluate(params, trial)
            return accuracy
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            raise optuna.TrialPruned()
    
    def _sample_params(self, trial: Trial) -> Dict[str, Any]:
        """Sample hyperparameters for DeepSeek."""
        return {
            'learning_rate': trial.suggest_float(
                'learning_rate',
                self.search_space.lr_min,
                self.search_space.lr_max,
                log=self.search_space.lr_log
            ),
            'batch_size': trial.suggest_categorical(
                'batch_size',
                self.search_space.batch_sizes
            ),
            'epochs': trial.suggest_int(
                'epochs',
                self.search_space.epochs_min,
                self.search_space.epochs_max
            ),
            'lora_r': trial.suggest_categorical(
                'lora_r',
                self.search_space.lora_r_options
            ),
            'lora_alpha': trial.suggest_categorical(
                'lora_alpha',
                self.search_space.lora_alpha_options
            ),
            'lora_dropout': trial.suggest_float(
                'lora_dropout',
                self.search_space.lora_dropout_min,
                self.search_space.lora_dropout_max
            ),
            'gradient_accumulation_steps': trial.suggest_categorical(
                'gradient_accumulation_steps',
                self.search_space.gradient_accumulation_options
            ),
            'weight_decay': trial.suggest_float(
                'weight_decay',
                self.search_space.weight_decay_min,
                self.search_space.weight_decay_max
            ),
            'warmup_ratio': trial.suggest_float(
                'warmup_ratio',
                self.search_space.warmup_ratio_min,
                self.search_space.warmup_ratio_max
            ),
        }
    
    def _train_and_evaluate(
        self,
        params: Dict[str, Any],
        trial: Trial,
    ) -> float:
        """Train DeepSeek model and return validation accuracy."""
        from src.vin_ocr.training.finetune_deepseek import (
            DeepSeekFineTuner,
            DeepSeekFineTuneConfig,
        )
        
        # Create training config
        train_config = DeepSeekFineTuneConfig(
            learning_rate=params['learning_rate'],
            batch_size=params['batch_size'],
            num_epochs=params['epochs'],
            lora_r=params['lora_r'],
            lora_alpha=params['lora_alpha'],
            lora_dropout=params['lora_dropout'],
            gradient_accumulation_steps=params['gradient_accumulation_steps'],
            weight_decay=params['weight_decay'],
            warmup_ratio=params['warmup_ratio'],
            # Data paths
            train_data_path=self.config.train_labels,
            val_data_path=self.config.val_labels,
            data_dir=self.config.train_data_dir,
            # Output
            output_dir=os.path.join(
                self.config.output_dir,
                f"trial_{trial.number}"
            ),
        )
        
        # Create trainer
        trainer = DeepSeekFineTuner(train_config)
        
        # Train with pruning callback
        def epoch_callback(epoch: int, metrics: Dict[str, float]):
            val_acc = metrics.get('val_accuracy', 0.0)
            trial.report(val_acc, epoch)
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        try:
            best_accuracy = trainer.train(epoch_callback=epoch_callback)
            return best_accuracy
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Training failed: {e}")
            return 0.0


# =============================================================================
# MAIN TUNER CLASS
# =============================================================================

class OptunaHyperparameterTuner:
    """
    Optuna-based hyperparameter tuner for VIN OCR models.
    
    Supports:
    - PaddleOCR models (PP-OCRv5, CRNN, SVTR_LCNet, SVTR_Tiny)
    - DeepSeek Vision-Language models
    - Multi-objective optimization
    - Distributed tuning
    - Result persistence
    
    Example:
        tuner = OptunaHyperparameterTuner(
            model_type='paddleocr',
            config=TuningConfig(n_trials=50)
        )
        best_params = tuner.optimize()
        print(f"Best params: {best_params}")
    """
    
    def __init__(
        self,
        model_type: str = 'paddleocr',
        config: Optional[TuningConfig] = None,
        search_space: Optional[Union[PaddleOCRSearchSpace, DeepSeekSearchSpace]] = None,
        device: str = "cpu",
    ):
        """
        Initialize the hyperparameter tuner.
        
        Args:
            model_type: 'paddleocr' or 'deepseek'
            config: Tuning configuration
            search_space: Custom search space (uses defaults if None)
            device: Device to train on ('cpu', 'cuda', 'mps')
        """
        self.model_type = model_type.lower()
        self.config = config or TuningConfig()
        self.device = device
        
        # Set up search space
        if search_space is not None:
            self.search_space = search_space
        elif self.model_type == 'paddleocr':
            self.search_space = PaddleOCRSearchSpace()
        elif self.model_type == 'deepseek':
            self.search_space = DeepSeekSearchSpace()
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Create output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Initialize study
        self.study = self._create_study()
        
        logger.info(f"OptunaHyperparameterTuner initialized for {model_type}")
    
    def _create_study(self) -> optuna.Study:
        """Create Optuna study with configured sampler and pruner."""
        # Sampler
        sampler = TPESampler(
            seed=self.config.sampler_seed,
            n_startup_trials=self.config.n_startup_trials,
        )
        
        # Pruner
        if self.config.enable_pruning:
            pruner = MedianPruner(
                n_startup_trials=self.config.pruning_warmup_trials,
                n_warmup_steps=self.config.pruning_warmup_steps,
            )
        else:
            pruner = optuna.pruners.NopPruner()
        
        # Create study
        study = optuna.create_study(
            study_name=self.config.study_name,
            storage=self.config.storage,
            direction=self.config.direction,
            sampler=sampler,
            pruner=pruner,
            load_if_exists=True,
        )
        
        return study
    
    def optimize(
        self,
        n_trials: Optional[int] = None,
        timeout: Optional[int] = None,
        callbacks: Optional[List[Callable]] = None,
    ) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            n_trials: Number of trials (overrides config)
            timeout: Timeout in seconds (overrides config)
            callbacks: Optional Optuna callbacks
            
        Returns:
            Best hyperparameters found
        """
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.timeout
        
        # Create objective function
        if self.model_type == 'paddleocr':
            objective = PaddleOCRObjective(
                self.search_space,
                self.config,
                self.device,
            )
        else:
            objective = DeepSeekObjective(
                self.search_space,
                self.config,
                self.device,
            )
        
        # Add early stopping callback
        all_callbacks = callbacks or []
        all_callbacks.append(self._early_stopping_callback)
        
        logger.info(f"Starting optimization with {n_trials} trials...")
        
        # Run optimization
        self.study.optimize(
            objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=self.config.n_jobs,
            callbacks=all_callbacks,
            show_progress_bar=True,
        )
        
        # Save results
        self._save_results()
        
        return self.get_best_params()
    
    def _early_stopping_callback(
        self,
        study: optuna.Study,
        trial: optuna.trial.FrozenTrial,
    ) -> None:
        """Early stopping callback."""
        if len(study.trials) < self.config.early_stopping_trials:
            return
        
        # Check if any trials have completed successfully
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        if not completed_trials:
            return  # No completed trials yet, can't determine early stopping
        
        # Check if there's been improvement in last N trials
        try:
            best_value = study.best_value
        except ValueError:
            return  # No best value yet
            
        recent_trials = study.trials[-self.config.early_stopping_trials:]
        recent_values = [t.value for t in recent_trials if t.value is not None]
        
        if not recent_values:
            return  # No recent completed trials
            
        recent_best = max(recent_values)
        
        if recent_best <= best_value * 0.99:  # No significant improvement
            logger.info("Early stopping: No improvement in recent trials")
            study.stop()
    
    def _save_results(self) -> None:
        """Save optimization results to disk."""
        # Check if any trials completed successfully
        completed_trials = [t for t in self.study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed_trials:
            results = {
                'study_name': self.config.study_name,
                'model_type': self.model_type,
                'n_trials': len(self.study.trials),
                'completed_trials': len(completed_trials),
                'best_trial': self.study.best_trial.number,
                'best_value': self.study.best_value,
                'best_params': self.study.best_params,
                'datetime': datetime.now().isoformat(),
            }
        else:
            # No trials completed - save what we can
            results = {
                'study_name': self.config.study_name,
                'model_type': self.model_type,
                'n_trials': len(self.study.trials),
                'completed_trials': 0,
                'best_trial': None,
                'best_value': None,
                'best_params': None,
                'datetime': datetime.now().isoformat(),
                'error': 'No trials completed successfully',
            }
            logger.warning("No trials completed successfully!")
        
        # Save JSON
        results_path = os.path.join(
            self.config.output_dir,
            'optimization_results.json'
        )
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Save trial history
        history_path = os.path.join(
            self.config.output_dir,
            'trial_history.csv'
        )
        df = self.study.trials_dataframe()
        df.to_csv(history_path, index=False)
        
        logger.info(f"Trial history saved to {history_path}")
    
    def get_best_params(self) -> Dict[str, Any]:
        """Get best hyperparameters found."""
        try:
            return self.study.best_params
        except ValueError:
            return {}
    
    def get_best_value(self) -> float:
        """Get best metric value achieved."""
        try:
            return self.study.best_value
        except ValueError:
            return 0.0
    
    def get_trial_history(self) -> List[Dict[str, Any]]:
        """Get history of all trials."""
        return [
            {
                'number': t.number,
                'value': t.value,
                'params': t.params,
                'state': t.state.name,
            }
            for t in self.study.trials
        ]
    
    def visualize(self, output_dir: Optional[str] = None) -> None:
        """
        Generate visualization plots.
        
        Requires: pip install optuna-dashboard plotly kaleido
        """
        output_dir = output_dir or self.config.output_dir
        
        try:
            from optuna.visualization import (
                plot_optimization_history,
                plot_param_importances,
                plot_parallel_coordinate,
                plot_slice,
            )
            
            # Optimization history
            fig = plot_optimization_history(self.study)
            fig.write_image(os.path.join(output_dir, 'optimization_history.png'))
            
            # Parameter importances
            if len(self.study.trials) >= 10:
                fig = plot_param_importances(self.study)
                fig.write_image(os.path.join(output_dir, 'param_importances.png'))
            
            # Parallel coordinate
            fig = plot_parallel_coordinate(self.study)
            fig.write_image(os.path.join(output_dir, 'parallel_coordinate.png'))
            
            logger.info(f"Visualizations saved to {output_dir}")
            
        except ImportError:
            logger.warning(
                "Visualization requires: pip install plotly kaleido"
            )


# =============================================================================
# CLI INTERFACE
# =============================================================================

def main():
    """Command-line interface for hyperparameter tuning."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Hyperparameter tuning for VIN OCR models'
    )
    parser.add_argument(
        '--model', '-m',
        choices=['paddleocr', 'deepseek'],
        default='paddleocr',
        help='Model type to tune'
    )
    parser.add_argument(
        '--n-trials', '-n',
        type=int,
        default=50,
        help='Number of trials'
    )
    parser.add_argument(
        '--timeout',
        type=int,
        default=None,
        help='Timeout in seconds'
    )
    parser.add_argument(
        '--train-data', '-t',
        default='./data/train',
        help='Training data directory'
    )
    parser.add_argument(
        '--train-labels',
        default='./data/train/labels.txt',
        help='Training labels file'
    )
    parser.add_argument(
        '--val-data',
        default='./data/val',
        help='Validation data directory'
    )
    parser.add_argument(
        '--val-labels',
        default='./data/val/labels.txt',
        help='Validation labels file'
    )
    parser.add_argument(
        '--output', '-o',
        default='./output/hyperparameter_tuning',
        help='Output directory'
    )
    parser.add_argument(
        '--device',
        default='cpu',
        choices=['cpu', 'cuda', 'mps'],
        help='Device to train on'
    )
    parser.add_argument(
        '--study-name',
        default='vin_ocr_tuning',
        help='Optuna study name'
    )
    parser.add_argument(
        '--storage',
        default=None,
        help='Optuna storage URL (e.g., sqlite:///tuning.db)'
    )
    
    args = parser.parse_args()
    
    # Create config
    config = TuningConfig(
        study_name=args.study_name,
        storage=args.storage,
        n_trials=args.n_trials,
        timeout=args.timeout,
        train_data_dir=args.train_data,
        train_labels=args.train_labels,
        val_data_dir=args.val_data,
        val_labels=args.val_labels,
        output_dir=args.output,
    )
    
    # Create tuner
    tuner = OptunaHyperparameterTuner(
        model_type=args.model,
        config=config,
        device=args.device,
    )
    
    # Run optimization
    best_params = tuner.optimize()
    
    print("\n" + "="*50)
    print("OPTIMIZATION COMPLETE")
    print("="*50)
    print(f"Best value: {tuner.get_best_value():.4f}")
    if best_params:
        print(f"Best parameters:")
        for k, v in best_params.items():
            print(f"  {k}: {v}")
    else:
        print("No trials completed successfully - check logs for errors")


if __name__ == '__main__':
    main()
