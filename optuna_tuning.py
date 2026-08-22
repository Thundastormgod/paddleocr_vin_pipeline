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
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional
import logging

from src.vin_ocr.tracking import start_run

# The single definition of "a trial produced no measurement", shared with the
# in-process tuner in src/vin_ocr/training/hyperparameter_tuning. Two
# same-named exception classes would drift exactly like this repository's
# duplicated charset maps and F1 scorers did - and an except clause written
# against one would silently not catch the other.
from src.vin_ocr.training.hyperparameter_tuning.errors import TrialExecutionError

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#: Repository root, derived from this file's location so the tuner works from
#: any working directory.
REPO_ROOT = Path(__file__).resolve().parent

#: Backward-compatible DEFAULT metrics location, used only when
#: run_training_trial is called without an explicit per-trial path.
#: The tuner itself derives the real location from the --base-config's
#: Global.save_model_dir (see _resolve_base_save_model_dir) and gives every
#: trial its own save_model_dir/trial_<n>/training_metrics.json.
#: Absolute, because the trial subprocess runs with cwd=REPO_ROOT while the
#: tuner may have been launched from anywhere. A relative path here read a
#: file in the tuner's CWD that the subprocess never wrote.
TRIAL_METRICS_PATH = REPO_ROOT / "output" / "vin_rec_finetune" / "training_metrics.json"

#: Wall-clock ceiling for a single training trial.
TRIAL_TIMEOUT_SECONDS = 3600


class VINOCRHyperparameterTuner:
    def __init__(self, base_config_path: str = "configs/vin_finetune_config.yml"):
        self.base_config_path = base_config_path
        self.results_dir = REPO_ROOT / "optuna_results"
        self.results_dir.mkdir(exist_ok=True)
        # Set here as well as in run_study so objective() is callable directly
        # (in tests, or from a custom driver) without an AttributeError.
        self.study_name: str = "vin_ocr_optimization"
        self.experiment_name: str = "vin_ocr_optimization"
        self.dataset_roots: tuple = ()
        # Where trial outputs live: derived from the base config's
        # Global.save_model_dir, NOT hardcoded (L35). Each trial then gets its
        # own trial_<n> subdirectory so checkpoints survive the study (L36).
        self.base_save_model_dir: Path = self._resolve_base_save_model_dir()
        logger.info(
            f"Trial output root: {self.base_save_model_dir} "
            f"(per trial: trial_<n>/training_metrics.json)"
        )

    def _resolve_config_path(self) -> Path:
        """Resolve the base config path against the repo root when relative."""
        cfg_path = Path(self.base_config_path)
        if not cfg_path.is_absolute():
            cfg_path = REPO_ROOT / cfg_path
        return cfg_path

    def _resolve_base_save_model_dir(self) -> Path:
        """
        Derive the trial output root from the base config (L35).

        Reads Global.save_model_dir from --base-config; falls back to the
        historical default (the parent of TRIAL_METRICS_PATH) when the config
        cannot be read or lacks the key, logging the resolved path either way.
        """
        default = TRIAL_METRICS_PATH.parent
        cfg_path = self._resolve_config_path()
        try:
            with open(cfg_path, 'r') as f:
                cfg = yaml.safe_load(f)
            save_dir = cfg['Global']['save_model_dir']
        except (OSError, yaml.YAMLError, KeyError, TypeError) as exc:
            logger.warning(
                f"Could not read Global.save_model_dir from {cfg_path} "
                f"({type(exc).__name__}: {exc}); falling back to {default}"
            )
            return default
        save_path = Path(str(save_dir))
        if not save_path.is_absolute():
            # Trial subprocesses run with cwd=REPO_ROOT, so a relative
            # save_model_dir resolves against the repo root.
            save_path = (REPO_ROOT / save_path).resolve()
        return save_path

    def trial_metrics_path(self, trial_number: int) -> Path:
        """Metrics file for one trial, inside that trial's own output dir."""
        return self.base_save_model_dir / f"trial_{trial_number}" / "training_metrics.json"

    def load_base_config(self) -> Dict[str, Any]:
        """Load the base configuration file."""
        with open(self._resolve_config_path(), 'r') as f:
            return yaml.safe_load(f)
    
    def create_trial_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Create a trial configuration with suggested hyperparameters."""
        config = self.load_base_config()
        
        # Learning rate. Upper bound is 2e-3, NOT 1e-2: measured on the
        # PP-OCRv4 architecture, a constant lr of 3e-3 drives training into
        # the CTC blank basin permanently (600 steps, loss pinned at
        # ln(34)=3.53, no escape) while 1e-3 learns. Sampling above ~2e-3
        # spends trial budget on configurations that cannot converge.
        config['Optimizer']['lr']['learning_rate'] = trial.suggest_float(
            'learning_rate', 0.0002, 0.002, log=True
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
        
        # Per-trial output directory (L36): with a shared save_model_dir every
        # trial overwrote the previous trial's checkpoints, so the best
        # trial's weights were unrecoverable at the end of the study.
        config['Global']['save_model_dir'] = str(
            self.base_save_model_dir / f"trial_{trial.number}"
        )
        
        return config
    
    def save_trial_config(self, config: Dict[str, Any], trial_number: int):
        """Save trial configuration for reproducibility."""
        trial_config_path = self.results_dir / f"trial_{trial_number}_config.yml"
        with open(trial_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
        return trial_config_path
    
    def run_training_trial(
        self, config_path: str, metrics_path: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Run a single training trial and return its measured metrics.

        Args:
            config_path: Trial configuration to train with.
            metrics_path: Where THIS trial's training_metrics.json will be
                written (the trial config's save_model_dir). Defaults to the
                module-level TRIAL_METRICS_PATH for backward compatibility
                with callers that share one output directory.

        Returns:
            Dict of measured metrics plus the wall-clock training time.

        Raises:
            TrialExecutionError: If training exits non-zero, times out, or
                fails to write fresh metrics. Every one of these is an absence
                of measurement and is reported as such.

        Note:
            This method previously scored a trial from
            ``output/vin_rec_finetune/training_metrics.json`` whenever that
            file merely EXISTED, without checking the subprocess exit status
            or when the file was written. Since every trial overwrites the same
            fixed path, a trial that crashed before writing was scored from the
            PREVIOUS trial's file - reporting another configuration's accuracy
            under this trial's hyperparameters, and returning it to Optuna as a
            genuine observation. Reproduced directly: after a successful trial
            recording 0.4186, a crashed trial returned 0.4186.

            Three independent guards now prevent that: the stale file is
            removed before launching, a non-zero exit is fatal, and the file's
            modification time must post-date the launch.
        """
        if metrics_path is None:
            metrics_path = TRIAL_METRICS_PATH

        # sys.executable (not bare "python") so the trial runs in the same
        # interpreter/venv as the tuner.
        cmd = [
            sys.executable, "-m", "src.vin_ocr.training.finetune_paddleocr",
            "--config", config_path
        ]

        # Guard 1: no output from a previous trial can be mistaken for this one.
        metrics_path.unlink(missing_ok=True)

        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=TRIAL_TIMEOUT_SECONDS,
                cwd=str(REPO_ROOT),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TrialExecutionError(
                f"training exceeded {TRIAL_TIMEOUT_SECONDS}s and was killed"
            ) from exc

        training_time = time.time() - start_time

        # Guard 2: a crash is a crash, whatever files happen to be on disk.
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-2000:]
            raise TrialExecutionError(
                f"training exited {result.returncode}:\n{tail or '<no output>'}"
            )

        if not metrics_path.is_file():
            raise TrialExecutionError(
                f"training exited 0 but wrote no metrics to {metrics_path}"
            )

        # Guard 3: the file must have been written by THIS run.
        if metrics_path.stat().st_mtime < start_time:
            raise TrialExecutionError(
                f"{metrics_path} predates this trial; it is a stale "
                f"artifact of an earlier run and must not be scored"
            )

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            image_level = metrics['evaluation_metrics']['image_level']
            char_level = metrics['evaluation_metrics']['character_level']
            training_results = metrics['training_results']
        except (json.JSONDecodeError, KeyError, OSError) as exc:
            raise TrialExecutionError(
                f"could not read metrics from {metrics_path}: {exc}"
            ) from exc

        return {
            'exact_match_accuracy': image_level['exact_match_accuracy'],
            'character_accuracy': char_level['character_accuracy'],
            'f1_micro': char_level['f1_micro'],
            'training_time': training_time,
            'final_epoch': training_results['final_epoch'],
            'best_validation_accuracy': training_results['best_validation_accuracy'],
        }
    
    def objective(self, trial: optuna.Trial) -> float:
        """
        Score one hyperparameter configuration.

        Each trial is recorded as its own tracked run carrying the commit, the
        working-tree diff, the resolved dependency versions and the dataset
        fingerprint, so any trial in the study can be replayed and checked. The
        63 JSON files this script previously left in optuna_results/ recorded
        hyperparameters and an accuracy with no commit, no data hash and no
        timestamp - enough to report a number, not enough to verify one.

        Args:
            trial: The Optuna trial supplying hyperparameters.

        Returns:
            Exact-match accuracy, the value being maximised.

        Raises:
            TrialExecutionError: If the trial produced no measurement. This
                propagates to study.optimize(catch=...), which records the
                trial as FAILED. It previously returned 0.0, which Optuna
                cannot distinguish from a real measurement of zero and which
                therefore poisoned the TPE surrogate model.
        """
        config = self.create_trial_config(trial)
        trial_number = trial.number
        config_path = self.save_trial_config(config, trial_number)
        logger.info(f"Trial {trial_number}: Starting with config {config_path}")

        with start_run(
            f"{self.study_name}-trial-{trial_number}",
            experiment=self.experiment_name,
            dataset_roots=self.dataset_roots,
            params=trial.params,
            tags={
                "study_name": self.study_name,
                "trial_number": trial_number,
                "optuna_sampler": "TPESampler",
            },
        ) as run:
            run.log_artifact(config_path, "trial_config")

            results = self.run_training_trial(
                str(config_path),
                metrics_path=self.trial_metrics_path(trial_number),
            )
            accuracy = results['exact_match_accuracy']

            run.log_metrics({
                'exact_match_accuracy': accuracy,
                'character_accuracy': results['character_accuracy'],
                'f1_micro': results['f1_micro'],
                'training_time_seconds': results['training_time'],
                'final_epoch': results['final_epoch'],
                'best_validation_accuracy': results['best_validation_accuracy'],
            })

            logger.info(
                f"Trial {trial_number}: Accuracy = {accuracy:.4f} "
                f"(run {run.run_id})"
            )

            trial_results = {
                'trial_number': trial_number,
                'params': trial.params,
                'accuracy': accuracy,
                'character_accuracy': results['character_accuracy'],
                'f1_micro': results['f1_micro'],
                'training_time': results['training_time'],
                'final_epoch': results['final_epoch'],
                # Provenance, so this file is self-describing even when read
                # outside MLflow. The existing trial_*_results.json files carry
                # none of this and cannot be traced to any commit.
                'mlflow_run_id': run.run_id,
                'git_commit': run.provenance.git.commit if run.provenance else None,
                'reproduce_command': run.reproduce_command,
            }

            results_path = self.results_dir / f"trial_{trial_number}_results.json"
            with open(results_path, 'w') as f:
                json.dump(trial_results, f, indent=2)

            run.log_artifact(results_path, "trial_results")
            return accuracy
    
    def run_study(
        self,
        n_trials: int = 50,
        study_name: str = "vin_ocr_optimization",
        experiment_name: Optional[str] = None,
        dataset_roots: tuple = (),
    ):
        """
        Run the Optuna study, resuming any existing study of the same name.

        Args:
            n_trials: Number of trials to run in this invocation.
            study_name: Study identifier, also the SQLite study key.
            experiment_name: MLflow experiment. Defaults to the study name.
            dataset_roots: Data directories to fingerprint into each run.

        Returns:
            The completed optuna.Study.

        Note:
            The study is persisted to SQLite. It previously had no ``storage``
            argument, so it lived only in memory: an interrupted study lost
            every completed trial and had to restart from scratch, and the
            sampler could not be warm-started. At up to one hour per trial and
            100 trials, that is days of compute discarded by a single Ctrl-C.
            ``load_if_exists`` makes re-running the command resume instead of
            colliding.
        """
        self.study_name = study_name
        self.experiment_name = experiment_name or study_name
        self.dataset_roots = tuple(dataset_roots)

        storage_path = self.results_dir / f"{study_name}.db"
        storage_uri = f"sqlite:///{storage_path}"

        logger.info(
            f"Starting Optuna study '{study_name}' with {n_trials} trials "
            f"(storage: {storage_uri})"
        )

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=42),
            study_name=study_name,
            storage=storage_uri,
            load_if_exists=True,
        )

        completed = [t for t in study.trials if t.value is not None]
        if completed:
            logger.info(f"Resuming: {len(completed)} completed trial(s) already recorded")

        def print_callback(study, trial):
            # M25: study.best_trial RAISES ValueError while no trial has
            # COMPLETED (`is not None` cannot guard that), and callbacks are
            # not covered by optimize(catch=...) - an exception here would
            # abort the whole study after a failed early trial. Guard on the
            # completed-trial count instead so failed trials are recorded
            # and skipped, never fatal.
            if trial.number % 5 != 0:
                return
            has_complete = any(
                t.state == optuna.trial.TrialState.COMPLETE for t in study.trials
            )
            if has_complete:
                logger.info(f"Trial {trial.number}: Best accuracy = {study.best_value:.4f}")

        # catch=(TrialExecutionError,) records a failed trial as FAILED rather
        # than aborting the whole study, while keeping it out of the sampler's
        # observations. Returning 0.0 instead - the previous behaviour - made a
        # crash indistinguishable from a measured zero.
        study.optimize(
            self.objective,
            n_trials=n_trials,
            callbacks=[print_callback],
            show_progress_bar=True,
            catch=(TrialExecutionError,),
        )
        
        # Separate real observations from failed trials. `study.trials` counts
        # both, so reporting len(study.trials) as "n_trials" overstated how
        # much was actually measured whenever a trial crashed.
        measured = [t for t in study.trials if t.value is not None]
        failed = len(study.trials) - len(measured)

        if not measured:
            # study.best_value raises here. Reporting nothing is correct;
            # inventing a best is what this codebase is being cleaned up for.
            logger.error(
                "OPTUNA STUDY COMPLETE - 0 of %d trials produced a measurement. "
                "No best configuration exists. Check the FAILED runs in MLflow "
                "for the training errors.",
                len(study.trials),
            )
            return study

        # L36: each trial trains into its own directory, so the best trial's
        # checkpoints are still on disk when the study ends.
        best_trial_dir = self.base_save_model_dir / f"trial_{study.best_trial.number}"

        best_results = {
            'study_name': study_name,
            'n_trials_total': len(study.trials),
            'n_trials_measured': len(measured),
            'n_trials_failed': failed,
            'best_trial': study.best_trial.number,
            'best_accuracy': study.best_value,
            'best_params': study.best_params,
            'best_trial_dir': str(best_trial_dir),
            'all_trials': [
                {
                    'trial_number': trial.number,
                    'accuracy': trial.value,
                    'params': trial.params,
                }
                for trial in measured
            ],
        }

        study_results_path = self.results_dir / f"{study_name}_results.json"
        with open(study_results_path, 'w') as f:
            json.dump(best_results, f, indent=2)

        logger.info("=" * 60)
        logger.info("OPTUNA STUDY COMPLETE")
        logger.info(f"Trials measured: {len(measured)} (failed: {failed})")
        logger.info(f"Best accuracy: {study.best_value:.4f}")
        logger.info(f"Best trial: {study.best_trial.number}")
        logger.info(f"Best trial checkpoints: {best_trial_dir}")
        logger.info("Best parameters:")
        for param, value in study.best_params.items():
            logger.info(f"  {param}: {value}")
        logger.info(f"Results saved to: {study_results_path}")
        logger.info("=" * 60)

        return study

def parse_args(argv=None):
    """Parse command-line arguments."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for VIN OCR fine-tuning.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=30,
        help="Trials to run in this invocation. The study resumes, so this is "
             "an increment, not a total. (default: 30)",
    )
    parser.add_argument(
        "--study-name", default="vin_ocr_comprehensive_tuning",
        help="Study name; also the SQLite study key used to resume.",
    )
    parser.add_argument(
        "--experiment", default=None,
        help="MLflow experiment name (default: the study name).",
    )
    parser.add_argument(
        "--dataset-root", action="append", default=[],
        help="Data directory to fingerprint into each run. Repeatable.",
    )
    parser.add_argument(
        "--base-config", default="configs/vin_finetune_config.yml",
        help="Base configuration each trial perturbs.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> int:
    """
    Run the hyperparameter search.

    Returns:
        0 on success, 1 when no trial produced a measurement.
    """
    args = parse_args(argv)
    tuner = VINOCRHyperparameterTuner(base_config_path=args.base_config)

    study = tuner.run_study(
        n_trials=args.n_trials,
        study_name=args.study_name,
        experiment_name=args.experiment,
        dataset_roots=tuple(Path(d) for d in args.dataset_root),
    )

    # Guard against the ValueError study.best_trial raises when every trial
    # failed. Emitting a "best config" from no measurements would be exactly
    # the class of fabricated artifact this codebase is being cleaned of.
    if not any(t.value is not None for t in study.trials):
        logger.error("No best configuration written: nothing was measured.")
        return 1

    best_config_path = tuner.results_dir / "best_config.yml"
    best_config = tuner.create_trial_config(study.best_trial)
    with open(best_config_path, 'w') as f:
        yaml.dump(best_config, f, default_flow_style=False)

    logger.info(f"Best configuration saved to: {best_config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
