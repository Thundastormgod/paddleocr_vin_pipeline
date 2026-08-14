#!/usr/bin/env python3
"""
Optuna Hyperparameter Tuning for VIN OCR
=====================================

This script uses Optuna to find the best hyperparameters for VIN OCR training.
It focuses on the most impactful parameters based on our analysis:

Key Parameters to Tune:
1. Learning Rate (0.0005 - 0.005)
2. Scheduler Type (CosineAnnealingDecay vs StepDecay)
3. Scheduler Parameters (T_max, step_decay_gamma)
4. Batch Size (8, 16, 32)
5. Optimizer Beta Values
6. Regularization Strength

Based on our findings:
- CosineAnnealingDecay performed best (4.65% vs 2.33%)
- Learning rate around 0.002 worked well
- Early stopping helps prevent overtraining
"""

import optuna
import yaml
import subprocess
import json
import time
from pathlib import Path
from typing import Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VINOCRHyperparameterTuner:
    def __init__(self, base_config_path: str = "configs/vin_finetune_config.yml"):
        self.base_config_path = base_config_path
        self.results_dir = Path("optuna_results")
        self.results_dir.mkdir(exist_ok=True)
        
    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file."""
        with open(self.base_config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def create_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create a trial configuration with suggested hyperparameters."""
        config = self.load_base_config()
        
        # Learning Rate Tuning - Expanded range based on best performance
        config['Optimizer']['lr']['learning_rate'] = trial.suggest_float(
            'learning_rate', 0.0005, 0.01, log=True
        )
        
        # Scheduler Type and Parameters
        scheduler_type = trial.suggest_categorical('scheduler_type', ['Cosine', 'CosineAnnealingDecay', 'StepDecay'])
        config['Optimizer']['lr']['name'] = scheduler_type
        
        if scheduler_type in ['Cosine', 'CosineAnnealingDecay']:
            config['Optimizer']['lr']['T_max'] = trial.suggest_int('T_max', 15, 35)
            config['Optimizer']['lr']['warmup_epoch'] = trial.suggest_int('warmup_epoch', 0, 10)
            config['Optimizer']['lr']['warmup_start_lr'] = trial.suggest_float(
                'warmup_start_lr', 1e-7, 1e-4, log=True
            )
            # Remove StepDecay specific params
            if 'step_decay_gamma' in config['Optimizer']['lr']:
                del config['Optimizer']['lr']['step_decay_gamma']
        else:  # StepDecay
            config['Optimizer']['lr']['step_decay_gamma'] = trial.suggest_float(
                'step_decay_gamma', 0.7, 0.95
            )
            # Remove Cosine specific params
            if 'T_max' in config['Optimizer']['lr']:
                del config['Optimizer']['lr']['T_max']
            if 'warmup_epoch' in config['Optimizer']['lr']:
                del config['Optimizer']['lr']['warmup_epoch']
            if 'warmup_start_lr' in config['Optimizer']['lr']:
                del config['Optimizer']['lr']['warmup_start_lr']
        
        # Optimizer Parameters
        config['Optimizer']['beta1'] = trial.suggest_float('beta1', 0.85, 0.95)
        config['Optimizer']['beta2'] = trial.suggest_float('beta2', 0.99, 0.999)
        
        # Regularization
        config['Optimizer']['regularizer']['factor'] = trial.suggest_float(
            'regularizer_factor', 1e-6, 1e-4, log=True
        )
        
        # Batch Size - Expanded options
        batch_size = trial.suggest_categorical('batch_size', [4, 8, 16, 32, 64])
        config['Train']['loader']['batch_size_per_card'] = batch_size
        config['Eval']['loader']['batch_size_per_card'] = batch_size
        
        # Early Stopping - Expanded range
        config['Global']['early_stopping_patience'] = trial.suggest_int('patience', 3, 20)
        config['Global']['early_stopping_min_delta'] = trial.suggest_float(
            'min_delta', 0.0001, 0.01, log=True
        )
        
        return config
    
    def save_trial_config(self, config: Dict[str, Any], trial_number: int):
        """Save trial configuration for reproducibility."""
        trial_config_path = self.results_dir / f"trial_{trial_number}_config.yml"
        with open(trial_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return trial_config_path
    
    def run_training_trial(self, config_path: str, max_epochs: int = 20) -> Dict[str, Any]:
        """Run a single training trial and return results."""
        try:
            # Run training with timeout
            cmd = [
                "python", "src/vin_ocr/training/finetune_paddleocr.py",
                "--config", config_path
            ]
            
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,  # 1 hour timeout
                cwd="/Users/startferanmi/Paddle/paddleocr_vin_pipeline"
            )
            training_time = time.time() - start_time
            
            # Parse results from training_metrics.json
            metrics_path = Path("output/vin_rec_finetune/training_metrics.json")
            if metrics_path.exists():
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                return {
                    'success': True,
                    'exact_match_accuracy': metrics['evaluation_metrics']['image_level']['exact_match_accuracy'],
                    'character_accuracy': metrics['evaluation_metrics']['character_level']['character_accuracy'],
                    'f1_micro': metrics['evaluation_metrics']['character_level']['f1_micro'],
                    'training_time': training_time,
                    'final_epoch': metrics['training_results']['final_epoch'],
                    'best_validation_accuracy': metrics['training_results']['best_validation_accuracy']
                }
            else:
                return {
                    'success': False,
                    'error': 'Training metrics not found',
                    'training_time': training_time,
                    'stdout': result.stdout,
                    'stderr': result.stderr
                }
                
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'error': 'Training timeout',
                'training_time': 3600
            }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'training_time': 0
            }
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization."""
        # Create trial configuration
        config = self.create_trial_config(trial)
        trial_number = trial.number
        
        # Save trial config
        config_path = self.save_trial_config(config, trial_number)
        logger.info(f"Trial {trial_number}: Starting with config {config_path}")
        
        # Run training
        results = self.run_training_trial(str(config_path))
        
        # Log results
        if results['success']:
            accuracy = results['exact_match_accuracy']
            logger.info(f"Trial {trial_number}: Accuracy = {accuracy:.4f}")
            
            # Save trial results
            trial_results = {
                'trial_number': trial_number,
                'params': trial.params,
                'accuracy': accuracy,
                'character_accuracy': results['character_accuracy'],
                'f1_micro': results['f1_micro'],
                'training_time': results['training_time'],
                'final_epoch': results['final_epoch']
            }
            
            results_path = self.results_dir / f"trial_{trial_number}_results.json"
            with open(results_path, 'w') as f:
                json.dump(trial_results, f, indent=2)
            
            return accuracy  # Maximize exact match accuracy
        else:
            logger.error(f"Trial {trial_number}: Failed - {results['error']}")
            return 0.0  # Penalty for failed trials
    
    def run_study(self, n_trials: int = 50, study_name: str = "vin_ocr_optimization"):
        """Run the Optuna study."""
        logger.info(f"Starting Optuna study: {study_name} with {n_trials} trials")
        
        # Create study with median sampler for balanced exploration
        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name
        )
        
        # Add callback for pruning
        def print_callback(study, trial):
            if trial.number % 5 == 0:
                logger.info(f"Trial {trial.number}: Best accuracy = {study.best_value:.4f}")
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[print_callback],
            show_progress_bar=True
        )
        
        # Save best results
        best_results = {
            'study_name': study_name,
            'n_trials': len(study.trials),
            'best_trial': study.best_trial.number,
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'all_trials': []
        }
        
        # Collect all trial results
        for trial in study.trials:
            if trial.value is not None:
                best_results['all_trials'].append({
                    'trial_number': trial.number,
                    'accuracy': trial.value,
                    'params': trial.params
                })
        
        # Save study results
        study_results_path = self.results_dir / f"{study_name}_results.json"
        with open(study_results_path, 'w') as f:
            json.dump(best_results, f, indent=2)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("OPTUNA STUDY COMPLETE")
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info("Best parameters:")
        for param, value in study.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Results saved to: {study_results_path}")
        logger.info("=" * 60)
        
        return study

def main():
    """Main function to run hyperparameter tuning."""
    tuner = VINOCRHyperparameterTuner()
    
    # Run study with expanded trials for thorough search
    study = tuner.run_study(
        n_trials=100,  # Increased from 30 to 100 for broader exploration
        study_name="vin_ocr_comprehensive_tuning"
    )
    
    # Create best config file
    best_config_path = tuner.results_dir / "best_config.yml"
    best_config = tuner.create_trial_config(study.best_trial)
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)
    
    logger.info(f"Best configuration saved to: {best_config_path}")

if __name__ == "__main__":
    main()
