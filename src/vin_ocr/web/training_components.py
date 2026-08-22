"""
Training UI Components for Streamlit
====================================

Provides training progress tracking and management for the web UI.

Components:
- TrainingState: Data class for training state
- ProgressTracker: Tracks training progress metrics
- TrainingRunner: Manages training processes
- TrainingUI: Renders training progress in Streamlit

Author: JRL-VIN Project
Date: February 2026
"""

try:
    import fcntl  # Unix-only
    _HAS_FCNTL = True
except ImportError:  # Windows
    fcntl = None
    _HAS_FCNTL = False
    try:
        import msvcrt  # Windows file locking
    except ImportError:
        msvcrt = None
import json
import os
import signal
import time
import threading
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import logging

import yaml

logger = logging.getLogger(__name__)

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


# =============================================================================
# PaddleOCR fine-tune config assembly
# =============================================================================
# finetune_paddleocr.py takes NO hyperparameter CLI flags — its argparse
# defines only --config/--resume/--export-onnx (plus optional DagsHub
# streaming flags). The ONLY way to carry UI settings into that trainer is a
# config YAML written for the run, passed via --config.

#: Base config consumed by finetune_paddleocr.py.
DEFAULT_FINETUNE_BASE_CONFIG = PROJECT_ROOT / "configs" / "vin_finetune_config.yml"

#: Architecture.algorithm values implemented by finetune_paddleocr's
#: VINFineTuner (see its SUPPORTED_ARCHITECTURES), keyed by the normalized
#: selector ("PP-OCRv5"/"PP_OCRv5"/"pp-ocrv5" all map to "ppocrv5").
FINETUNE_ALGORITHM_MAP = {
    "ppocrv4": "PP-OCRv4",
    "ppocrv5": "PP-OCRv5",
    "rosetta": "Rosetta",
}


def _normalize_architecture(architecture: str) -> str:
    """Lowercase and drop separators: 'PP-OCRv5' / 'PP_OCRv5' -> 'ppocrv5'."""
    return "".join(c for c in architecture.lower() if c.isalnum())


def _load_base_finetune_config(base_config_path: Optional[str]) -> Tuple[Dict[str, Any], Path]:
    """Load the base fine-tune config YAML, resolving against PROJECT_ROOT."""
    base_path = Path(base_config_path) if base_config_path else DEFAULT_FINETUNE_BASE_CONFIG
    if not base_path.is_absolute():
        base_path = PROJECT_ROOT / base_path
    if not base_path.exists():
        raise FileNotFoundError(f"Base fine-tune config not found: {base_path}")
    with open(base_path, "r") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Base fine-tune config is not a mapping: {base_path}")
    return config, base_path


def _resolve_dict_paths(config: Dict[str, Any]) -> None:
    """
    Make character_dict_path entries absolute (against PROJECT_ROOT).

    The merged config is written into the run's output directory; relative
    dictionary paths would then depend on the trainer's CWD.
    """
    for section_name in ("Global", "PostProcess"):
        section = config.get(section_name)
        if not isinstance(section, dict):
            continue
        dict_path = section.get("character_dict_path")
        if isinstance(dict_path, str) and dict_path and not Path(dict_path).is_absolute():
            section["character_dict_path"] = str(PROJECT_ROOT / dict_path)


def build_finetune_config(
    ui_config: Dict[str, Any],
    output_dir: str,
    base_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Merge UI training settings into the finetune_paddleocr base config.

    Key mapping (UI -> config keys the trainer actually reads, per its
    _REQUIRED_CONFIG_KEYS):
        epochs        -> Global.epoch_num (and Optimizer.lr.T_max for Cosine,
                         which the base config defines as the cosine period
                         in epochs)
        batch_size    -> Train.loader.batch_size_per_card
        learning_rate -> Optimizer.lr.learning_rate
        output_dir    -> Global.save_model_dir
        device        -> Global.use_gpu (False for 'cpu', True otherwise)
        train_data_dir / train_labels -> Train.dataset.data_dir / label_file_list
        val_data_dir / val_labels     -> Eval.dataset.data_dir / label_file_list
        architecture  -> Architecture.algorithm (only values the trainer
                         implements: PP-OCRv4, PP-OCRv5, Rosetta)

    Args:
        ui_config: Settings collected by the web UI.
        output_dir: Run output directory (becomes Global.save_model_dir).
        base_config_path: Base YAML; defaults to configs/vin_finetune_config.yml.

    Returns:
        The merged config dict, ready to be written with write_finetune_config.

    Raises:
        FileNotFoundError: The base config does not exist.
        ValueError: The base config is not a mapping, or the requested
            architecture is not implemented by the trainer.
    """
    merged, base_path = _load_base_finetune_config(base_config_path)

    def section(*path_keys: str) -> Dict[str, Any]:
        node: Dict[str, Any] = merged
        for key in path_keys:
            child = node.get(key)
            if not isinstance(child, dict):
                child = {}
                node[key] = child
            node = child
        return node

    global_cfg = section("Global")
    global_cfg["save_model_dir"] = str(output_dir)
    global_cfg["use_gpu"] = str(ui_config.get("device", "cpu")).lower() != "cpu"
    if "epochs" in ui_config:
        global_cfg["epoch_num"] = int(ui_config["epochs"])
        lr_cfg = section("Optimizer", "lr")
        if str(lr_cfg.get("name", "")).lower() == "cosine" and "T_max" in lr_cfg:
            lr_cfg["T_max"] = int(ui_config["epochs"])
    if "learning_rate" in ui_config:
        section("Optimizer", "lr")["learning_rate"] = float(ui_config["learning_rate"])
    if "batch_size" in ui_config:
        section("Train", "loader")["batch_size_per_card"] = int(ui_config["batch_size"])

    dataset_overrides = (
        ("train_data_dir", ("Train", "dataset"), "data_dir", False),
        ("train_labels", ("Train", "dataset"), "label_file_list", True),
        ("val_data_dir", ("Eval", "dataset"), "data_dir", False),
        ("val_labels", ("Eval", "dataset"), "label_file_list", True),
    )
    for ui_key, section_path, config_key, as_list in dataset_overrides:
        value = ui_config.get(ui_key)
        if value:
            section(*section_path)[config_key] = [str(value)] if as_list else str(value)

    architecture = str(ui_config.get("architecture") or "")
    if architecture:
        algorithm = FINETUNE_ALGORITHM_MAP.get(_normalize_architecture(architecture))
        if algorithm is None:
            supported = ", ".join(sorted(set(FINETUNE_ALGORITHM_MAP.values())))
            raise ValueError(
                f"Architecture {architecture!r} is not implemented by "
                f"finetune_paddleocr.py (supported: {supported}); the "
                f"architecture is otherwise fixed by the base config "
                f"({base_path.name})."
            )
        section("Architecture")["algorithm"] = algorithm

    _resolve_dict_paths(merged)
    return merged


def write_finetune_config(config: Dict[str, Any], output_dir: str) -> Path:
    """
    Write the merged trainer config into the run's output directory.

    Returns:
        Absolute path of the written YAML (passed to the trainer's --config).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    config_path = out / "train_config.yml"
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)
    return config_path


@dataclass
class TrainingUpdate:
    """Single training update event."""
    timestamp: str
    epoch: int
    batch: int
    loss: float
    accuracy: float = 0.0
    message: str = ""


@dataclass
class TrainingState:
    """Current state of training."""
    is_running: bool = False
    is_paused: bool = False
    current_epoch: int = 0
    total_epochs: int = 0
    current_batch: int = 0
    total_batches: int = 0
    current_loss: float = 0.0
    best_accuracy: float = 0.0
    start_time: Optional[float] = None
    error: Optional[str] = None
    #: Distinct terminal status: the user aborted the run. A stopped run is
    #: neither "completed" nor "failed" and the UI must render it as such.
    stopped: bool = False
    history: List[TrainingUpdate] = field(default_factory=list)


class ProgressTracker:
    """
    Tracks training progress and provides formatted status.
    
    Thread-safe singleton for sharing state across Streamlit reruns.
    """
    
    _instance: Optional['ProgressTracker'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._state = TrainingState()
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._state = TrainingState()
            self._initialized = True
    
    def get_state(self) -> TrainingState:
        """Get current training state."""
        return self._state
    
    def start(self, total_epochs: int, total_batches: int = 100):
        """Mark training as started."""
        self._state.is_running = True
        self._state.is_paused = False
        self._state.total_epochs = total_epochs
        self._state.total_batches = total_batches
        self._state.current_epoch = 1
        self._state.current_batch = 0
        self._state.start_time = time.time()
        self._state.error = None
        self._state.stopped = False
        self._state.history = []
    
    def update(self, epoch: int, batch: int, loss: float, accuracy: float = 0.0, message: str = ""):
        """Record a training update."""
        self._state.current_epoch = epoch
        self._state.current_batch = batch
        self._state.current_loss = loss
        
        if accuracy > self._state.best_accuracy:
            self._state.best_accuracy = accuracy
        
        update = TrainingUpdate(
            timestamp=datetime.now().isoformat(),
            epoch=epoch,
            batch=batch,
            loss=loss,
            accuracy=accuracy,
            message=message,
        )
        self._state.history.append(update)
        
        # Keep only last 1000 updates
        if len(self._state.history) > 1000:
            self._state.history = self._state.history[-500:]
    
    def complete(self, message: str = "Training completed"):
        """Mark training as completed."""
        self._state.is_running = False
        self.update(
            self._state.current_epoch,
            self._state.total_batches,
            self._state.current_loss,
            self._state.best_accuracy,
            message,
        )
    
    def error(self, error_message: str):
        """Mark training as failed."""
        self._state.is_running = False
        self._state.error = error_message
    
    def mark_stopped(self, message: str = "Training stopped by user"):
        """
        Mark training as stopped by the user (distinct terminal status).
        
        A user-aborted run is neither completed nor failed: `stopped` is set,
        `error` is cleared, and the message lands in the history for display.
        """
        self._state.is_running = False
        self._state.stopped = True
        self._state.error = None
        self.update(
            self._state.current_epoch,
            self._state.current_batch,
            self._state.current_loss,
            self._state.best_accuracy,
            message,
        )
    
    def pause(self):
        """Pause training."""
        self._state.is_paused = True
    
    def resume(self):
        """Resume training."""
        self._state.is_paused = False
    
    def reset(self):
        """Reset state."""
        self._state = TrainingState()
    
    def format_elapsed_time(self) -> str:
        """Format elapsed time as HH:MM:SS."""
        if self._state.start_time is None:
            return "00:00:00"
        
        elapsed = time.time() - self._state.start_time
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def format_remaining_time(self) -> str:
        """Estimate remaining time based on progress."""
        if self._state.start_time is None or not self._state.is_running:
            return "--:--:--"
        
        total_steps = self._state.total_epochs * self._state.total_batches
        current_step = (self._state.current_epoch - 1) * self._state.total_batches + self._state.current_batch
        
        if current_step == 0:
            return "--:--:--"
        
        elapsed = time.time() - self._state.start_time
        rate = current_step / elapsed
        remaining_steps = total_steps - current_step
        remaining_seconds = remaining_steps / rate if rate > 0 else 0
        
        hours, remainder = divmod(int(remaining_seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class TrainingRunner:
    """
    Manages training processes.
    
    Runs training in a subprocess and monitors progress.
    Thread-safe singleton with file-based lock to prevent simultaneous training.
    """
    
    _instance: Optional['TrainingRunner'] = None
    _lock = threading.Lock()
    # Anchored to PROJECT_ROOT like every other path this module resolves:
    # a CWD-relative lock file silently stops being a mutex the moment the
    # server is launched from a different directory (W-M13).
    _lock_file = PROJECT_ROOT / "output" / ".training_lock"
    
    # Training parameter bounds for validation
    PARAM_BOUNDS = {
        "epochs": (1, 1000),
        "batch_size": (1, 256),
        "learning_rate": (1e-8, 1.0),
        "n_trials": (1, 1000),
    }
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._process = None
                    cls._instance._tracker = get_global_tracker()
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._process: Optional[subprocess.Popen] = None
            self._tracker = get_global_tracker()
            self._stop_requested = False
            self._current_model_type = None  # Track which model is training
            self._current_training_type = None  # e.g. "paddleocr_finetune"
            self._current_console_log = None  # Track current console log path
            self._monitor_thread: Optional[threading.Thread] = None
            self._lock_fd: Optional[int] = None  # File descriptor for fcntl lock
            self._initialized = True
    
    def _validate_training_config(self, config: Dict[str, Any], training_type: str) -> None:
        """
        Validate training configuration before starting subprocess.
        
        Raises:
            ValueError: If configuration is invalid
        """
        errors = []
        
        # Validate numeric parameters
        for param, (min_val, max_val) in self.PARAM_BOUNDS.items():
            if param in config:
                value = config[param]
                if not isinstance(value, (int, float)):
                    errors.append(f"Invalid {param}: must be a number, got {type(value).__name__}")
                elif value < min_val or value > max_val:
                    errors.append(f"Invalid {param}: {value} (must be {min_val}-{max_val})")
        
        # Helper to resolve paths - check both as-is and relative to project root
        def resolve_path(path_str: str) -> Optional[Path]:
            """Resolve a path, checking both cwd and project root."""
            if not path_str:
                return None
            p = Path(path_str)
            if p.exists():
                return p
            # Try relative to project root
            p_from_root = PROJECT_ROOT / path_str
            if p_from_root.exists():
                return p_from_root
            return None
        
        # Validate required paths for training (not hyperparameter tuning)
        if "tuning" not in training_type:
            train_labels = config.get("train_labels")
            if train_labels:
                resolved = resolve_path(train_labels)
                if not resolved:
                    errors.append(f"Training labels file not found: {train_labels}")
                else:
                    logger.info(f"Resolved train labels: {resolved}")
            
            val_labels = config.get("val_labels")
            if val_labels:
                resolved = resolve_path(val_labels)
                if not resolved:
                    # Warning only - validation set is optional for some training
                    logger.warning(f"Validation labels file not found: {val_labels}")
                else:
                    logger.info(f"Resolved val labels: {resolved}")
        
        # Validate device selection. A device string is valid when it names
        # one of the supported device tokens (display forms like
        # "GPU (CUDA - Paddle)" or "MPS (Apple Silicon)" contain one).
        device = config.get("device", "").lower()
        valid_devices = ("cpu", "cuda", "mps", "gpu")
        if device and not any(token in device for token in valid_devices):
            errors.append(f"Invalid device: {device} (expected one of {', '.join(valid_devices)})")
        
        if errors:
            raise ValueError("Invalid training configuration:\n  - " + "\n  - ".join(errors))
    
    def _build_output_dir(self, base_dir: str, model_tag: str) -> str:
        """
        Create output directory for training.
        
        If the base_dir already contains a custom name (not default pattern), 
        use it directly. Otherwise, create a timestamped directory.
        
        Args:
            base_dir: Base directory path (e.g., "./output/my_custom_model" or "./output/vin_rec_finetune")
            model_tag: Model identifier tag (e.g., "paddleocr_finetune", "deepseek_finetune")
        
        Returns:
            Absolute directory path
        """
        base_path = Path(base_dir)
        
        # Resolve relative paths to absolute using PROJECT_ROOT
        if not base_path.is_absolute():
            base_path = PROJECT_ROOT / base_dir
        
        # Check if user provided a custom name (not a default pattern)
        dir_name = base_path.name
        default_patterns = ['vin_rec_finetune', 'deepseek_finetune', 'hyperparameter_tuning', 
                           'paddleocr_finetune', 'deepseek_scratch']
        
        # If it's a custom name, use it directly
        if dir_name not in default_patterns:
            return str(base_path)
        
        # Otherwise, create timestamped directory with model tag
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        parent_dir = base_path.parent
        new_dir = parent_dir / f"{model_tag}_{timestamp}"
        return str(new_dir)
    
    def _build_console_log_path(self, output_dir: str, model_tag: str) -> str:
        """
        Create model-specific console log file path.
        
        Args:
            output_dir: Output directory for this training run
            model_tag: Model identifier tag (e.g., "paddleocr", "deepseek")
        
        Returns:
            Console log file path (e.g., "./output/paddleocr_finetune_20260202_143022/paddleocr_console.log")
        """
        output_path = Path(output_dir)
        return str(output_path / f"{model_tag}_console.log")
    
    def get_current_model_type(self) -> Optional[str]:
        """Get the model type currently being trained."""
        return self._current_model_type
    
    def get_current_console_log(self) -> Optional[str]:
        """Get the console log path for the current training."""
        return self._current_console_log
    
    @property
    def is_running(self) -> bool:
        """Check if training is running (also checks lock file for cross-session detection)."""
        # Check subprocess
        if self._process is not None and self._process.poll() is None:
            return True
        
        # Check lock file for training started by another session
        if self._lock_file.exists():
            try:
                with open(self._lock_file, 'r') as f:
                    lock_data = json.load(f)
                pid = lock_data.get('pid')
                # Check if process is still running
                if pid and self._pid_alive(int(pid)):
                    return True
                # Process not running (or no pid recorded): stale lock
                self._lock_file.unlink(missing_ok=True)
            except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
                # Malformed/unreadable lock state must not crash the status
                # check, but it must not vanish silently either (W-L12).
                logger.debug(f"Could not parse training lock {self._lock_file}: {e}")
        
        return False
    
    def _acquire_lock(self, training_type: str, output_dir: str) -> bool:
        """
        Acquire exclusive lock using OS-specific file locking.
        
        This implementation prevents the TOCTOU race condition by using
        kernel-level file locking instead of check-then-create pattern.
        
        Returns:
            True if lock acquired, raises RuntimeError if already locked.
        """
        self._lock_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Open or create lock file
        try:
            self._lock_fd = os.open(str(self._lock_file), os.O_RDWR | os.O_CREAT)
        except OSError as e:
            raise RuntimeError(f"Cannot create lock file: {e}")
        
        try:
            # Try to acquire exclusive lock (non-blocking)
            if _HAS_FCNTL:
                fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:
                # Lock 1 byte at start of file
                os.lseek(self._lock_fd, 0, os.SEEK_SET)
                msvcrt.locking(self._lock_fd, msvcrt.LK_NBLCK, 1)
            else:
                raise RuntimeError("File locking not supported on this platform")
        except (BlockingIOError, OSError):
            # Lock held by another process - try to read who has it
            os.close(self._lock_fd)
            self._lock_fd = None
            
            try:
                with open(self._lock_file, 'r') as f:
                    lock_data = json.load(f)
                    existing_type = lock_data.get('training_type', 'unknown')
                    existing_pid = lock_data.get('pid', 'unknown')
                    raise RuntimeError(
                        f"Training already in progress: {existing_type} (PID {existing_pid}). "
                        "Please wait for it to complete or stop it first."
                    )
            except (json.JSONDecodeError, FileNotFoundError):
                raise RuntimeError("Training lock held by another process. Please try again.")
        
        # We have the lock - write our info
        lock_data = {
            'pid': os.getpid(),
            'training_type': training_type,
            'output_dir': output_dir,
            'started': datetime.now().isoformat(),
        }
        
        # Truncate and write lock info
        os.ftruncate(self._lock_fd, 0)
        os.lseek(self._lock_fd, 0, os.SEEK_SET)
        os.write(self._lock_fd, json.dumps(lock_data).encode())
        
        return True
    
    def _release_lock(self):
        """Release the file lock and remove lock file."""
        try:
            if hasattr(self, '_lock_fd') and self._lock_fd is not None:
                # Release lock
                if _HAS_FCNTL:
                    fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
                elif msvcrt is not None:
                    os.lseek(self._lock_fd, 0, os.SEEK_SET)
                    msvcrt.locking(self._lock_fd, msvcrt.LK_UNLCK, 1)
                os.close(self._lock_fd)
                self._lock_fd = None
            
            # Remove lock file
            if self._lock_file.exists():
                self._lock_file.unlink()
        except Exception as e:
            logger.warning(f"Error releasing lock: {e}")
    
    def start_paddleocr_finetuning(self, config: Dict[str, Any]):
        """
        Start the PaddleOCR fine-tuning process.

        finetune_paddleocr.py accepts no hyperparameter flags — its argparse
        defines only --config/--resume/--export-onnx. The UI settings are
        merged into the base config YAML (build_finetune_config), the merged
        config is written into the run's output directory, and the trainer is
        launched with --config pointing at that file.

        Raises:
            RuntimeError: Training already in progress.
            ValueError: Invalid settings, or an architecture the trainer
                does not implement.
            FileNotFoundError: Base config missing.
        """
        if self.is_running:
            raise RuntimeError("Training already in progress. Please wait for it to complete or stop it first.")
        
        # Validate configuration before starting
        self._validate_training_config(config, "paddleocr_finetune")
        
        # Create timestamped, model-specific output directory
        base_output_dir = config.get("output_dir", "./output/vin_rec_finetune")
        output_dir = self._build_output_dir(base_output_dir, "paddleocr_finetune")
        
        # Build the merged trainer config BEFORE touching tracker/lock state:
        # an unsupported architecture or a missing base config must surface
        # without leaving a phantom "running" tracker or a stale lock.
        merged_config = build_finetune_config(
            ui_config=config,
            output_dir=output_dir,
            base_config_path=config.get("base_config"),
        )
        
        self._tracker.start(config.get("epochs", 10))
        self._stop_requested = False
        
        # Set current model type for UI from the algorithm that will actually
        # be trained (the merged config is the single source of truth).
        self._current_model_type = merged_config.get("Architecture", {}).get(
            "algorithm", "PaddleOCR"
        )
        self._current_training_type = "paddleocr_finetune"
        
        # Acquire lock to prevent simultaneous training
        self._acquire_lock("paddleocr_finetune", output_dir)
        
        try:
            config_path = write_finetune_config(merged_config, output_dir)
            
            cmd = [
                sys.executable, "-m", "src.vin_ocr.training.finetune_paddleocr",
                "--config", str(config_path),
            ]
            
            logger.info(f"✓ Starting PaddleOCR Fine-Tuning")
            logger.info(f"  Model: {self._current_model_type}")
            logger.info(f"  Training Script: src.vin_ocr.training.finetune_paddleocr")
            logger.info(f"  Device: {'GPU' if merged_config['Global'].get('use_gpu') else 'CPU'}")
            logger.info(f"  Output Directory: {output_dir}")
            logger.info(f"  Config: {config_path}")
            logger.info(f"  Command: {' '.join(cmd)}")
            
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(PROJECT_ROOT),
            )
            
            # Start monitoring thread with output directory and model tag
            thread = threading.Thread(
                target=self._monitor_process, 
                args=(output_dir, "paddleocr"),
                daemon=True
            )
            self._monitor_thread = thread
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
            self._current_training_type = None
            self._release_lock()  # Release lock on error
            raise
    
    def start_deepseek_finetuning(self, config: Dict[str, Any]):
        """Start DeepSeek fine-tuning process."""
        if self.is_running:
            raise RuntimeError("Training already in progress. Please wait for it to complete or stop it first.")
        
        # Validate configuration before starting
        self._validate_training_config(config, "deepseek_finetune")
        
        self._tracker.start(config.get("epochs", 10))
        self._stop_requested = False
        
        # Create timestamped, model-specific output directory
        base_output_dir = config.get("output_dir", "./output/deepseek_finetune")
        output_dir = self._build_output_dir(base_output_dir, "deepseek_finetune")
        
        # Set current model type for UI - DeepSeek VL2
        self._current_model_type = "DeepSeek-VL2"
        self._current_training_type = "deepseek_finetune"
        
        # Acquire lock to prevent simultaneous training
        self._acquire_lock("deepseek_finetune", output_dir)
        
        cmd = [
            sys.executable, "-m", "src.vin_ocr.training.finetune_deepseek",
            "--epochs", str(config.get("epochs", 10)),
            "--batch-size", str(config.get("batch_size", 2)),
            "--lr", str(config.get("learning_rate", 0.00002)),
            "--output", output_dir,
        ]
        
        # Add LoRA flag if enabled
        if config.get("use_lora", True):
            cmd.append("--lora")
        else:
            cmd.append("--full")
        
        # Add data paths if provided
        if config.get("train_data_path"):
            cmd.extend(["--train-data", config.get("train_data_path")])
        if config.get("val_data_path"):
            cmd.extend(["--val-data", config.get("val_data_path")])
        if config.get("data_dir"):
            cmd.extend(["--data-dir", config.get("data_dir")])
        
        logger.info(f"✓ Starting DeepSeek Fine-Tuning")
        logger.info(f"  Model: DeepSeek-OCR")
        logger.info(f"  Training Script: src.vin_ocr.training.finetune_deepseek")
        logger.info(f"  Output Directory: {output_dir}")
        logger.info(f"  Command: {' '.join(cmd)}")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent.parent.parent.parent),  # Project root
            )
            
            # Start monitoring thread with output directory and model tag
            thread = threading.Thread(
                target=self._monitor_process,
                args=(output_dir, "deepseek"),
                daemon=True
            )
            self._monitor_thread = thread
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
            self._current_training_type = None
            self._release_lock()  # Release lock on error
            raise
    
    def start_paddleocr_scratch(self, config: Dict[str, Any]):
        """Start PaddleOCR from-scratch training."""
        if self.is_running:
            raise RuntimeError("Training already in progress. Please wait for it to complete or stop it first.")
        
        # Validate configuration before starting
        self._validate_training_config(config, "paddleocr_scratch")
        
        self._tracker.start(config.get("epochs", 100))
        self._stop_requested = False
        
        # Create timestamped, model-specific output directory
        base_output_dir = config.get("output_dir", "./output/vin_scratch")
        output_dir = self._build_output_dir(base_output_dir, "paddleocr_scratch")
        
        # Set current model type for UI - include architecture if specified
        architecture = config.get("architecture", "")
        if architecture:
            self._current_model_type = f"{architecture} (Scratch)"
        else:
            self._current_model_type = "PaddleOCR (Scratch)"
        self._current_training_type = "paddleocr_scratch"
        
        # Acquire lock to prevent simultaneous training
        self._acquire_lock("paddleocr_scratch", output_dir)
        
        cmd = [
            sys.executable, "-m", "src.vin_ocr.training.train_from_scratch",
            "--model", "paddleocr",
            "--epochs", str(config.get("epochs", 100)),
            "--batch-size", str(config.get("batch_size", 64)),
            "--lr", str(config.get("learning_rate", 0.001)),
            "--output-dir", output_dir,
        ]
        
        # Add data paths
        if config.get("train_data_dir"):
            cmd.extend(["--train-data-dir", config.get("train_data_dir")])
        if config.get("train_labels"):
            cmd.extend(["--train-labels", config.get("train_labels")])
        if config.get("val_data_dir"):
            cmd.extend(["--val-data-dir", config.get("val_data_dir")])
        if config.get("val_labels"):
            cmd.extend(["--val-labels", config.get("val_labels")])
        
        # Add architecture
        if config.get("architecture"):
            cmd.extend(["--architecture", config.get("architecture")])
        
        # Add device
        if config.get("device"):
            cmd.extend(["--device", config.get("device")])
        
        logger.info(f"✓ Starting PaddleOCR Training from Scratch")
        logger.info(f"  Model: PaddleOCR")
        logger.info(f"  Training Script: src.vin_ocr.training.train_from_scratch")
        logger.info(f"  Architecture: {config.get('architecture', 'default')}")
        logger.info(f"  Output Directory: {output_dir}")
        logger.info(f"  Command: {' '.join(cmd)}")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent.parent.parent.parent),  # Project root
            )
            
            # Start monitoring thread with output directory and model tag
            thread = threading.Thread(
                target=self._monitor_process,
                args=(output_dir, "paddleocr"),
                daemon=True
            )
            self._monitor_thread = thread
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
            self._current_training_type = None
            self._release_lock()  # Release lock on error
            raise
    
    def start_deepseek_scratch(self, config: Dict[str, Any]):
        """Start DeepSeek from-scratch training."""
        if self.is_running:
            raise RuntimeError("Training already in progress. Please wait for it to complete or stop it first.")
        
        # Validate configuration before starting
        self._validate_training_config(config, "deepseek_scratch")
        
        self._tracker.start(config.get("epochs", 50))
        self._stop_requested = False
        
        # Create timestamped, model-specific output directory
        base_output_dir = config.get("output_dir", "./output/deepseek_scratch")
        output_dir = self._build_output_dir(base_output_dir, "deepseek_scratch")
        
        # Set current model type for UI - DeepSeek-VL2 from Scratch
        self._current_model_type = "DeepSeek-VL2 (Scratch)"
        self._current_training_type = "deepseek_scratch"
        
        # Acquire lock to prevent simultaneous training
        self._acquire_lock("deepseek_scratch", output_dir)
        
        cmd = [
            sys.executable, "-m", "src.vin_ocr.training.train_from_scratch",
            "--model", "deepseek",
            "--epochs", str(config.get("epochs", 50)),
            "--batch-size", str(config.get("batch_size", 4)),
            "--lr", str(config.get("learning_rate", 0.0001)),
            "--output-dir", output_dir,
        ]
        
        # Add data paths
        if config.get("train_data_dir"):
            cmd.extend(["--train-data-dir", config.get("train_data_dir")])
        if config.get("train_labels"):
            cmd.extend(["--train-labels", config.get("train_labels")])
        if config.get("val_labels"):
            cmd.extend(["--val-labels", config.get("val_labels")])
        
        # Add device
        if config.get("device"):
            cmd.extend(["--device", config.get("device")])
        
        logger.info(f"✓ Starting DeepSeek Training from Scratch")
        logger.info(f"  Model: DeepSeek-OCR")
        logger.info(f"  Training Script: src.vin_ocr.training.train_from_scratch")
        logger.info(f"  Output Directory: {output_dir}")
        logger.info(f"  Command: {' '.join(cmd)}")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent.parent.parent.parent),  # Project root
            )
            
            # Start monitoring thread with output directory and model tag
            thread = threading.Thread(
                target=self._monitor_process,
                args=(output_dir, "deepseek"),
                daemon=True
            )
            self._monitor_thread = thread
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
            self._current_training_type = None
            self._release_lock()  # Release lock on error
            raise
    
    def start_hyperparameter_tuning(self, config: Dict[str, Any]):
        """
        Start Optuna hyperparameter tuning process.
        
        Args:
            config: Configuration dict with:
                - model_type: 'paddleocr' or 'deepseek'
                - n_trials: Number of optimization trials
                - train_data_dir: Training data directory
                - train_labels: Training labels file
                - val_labels: Validation labels file
                - output_dir: Output directory for results
                - device: 'cpu', 'cuda', or 'mps'
                - timeout: Optional timeout in seconds
        """
        if self.is_running:
            raise RuntimeError("Training already in progress. Please wait for it to complete or stop it first.")
        
        # Validate configuration before starting
        self._validate_training_config(config, "hyperparameter_tuning")
        
        self._tracker.start(config.get("n_trials", 50))
        self._stop_requested = False
        
        # Create timestamped, model-specific output directory
        model_type = config.get("model_type", "paddleocr")
        base_output_dir = config.get("output_dir", "./output/hyperparameter_tuning")
        output_dir = self._build_output_dir(base_output_dir, f"{model_type}_tuning")
        
        # Set current model type for UI - show friendly name
        if model_type == "paddleocr":
            self._current_model_type = "PaddleOCR (HP Tuning)"
        elif model_type == "deepseek":
            self._current_model_type = "DeepSeek-VL2 (HP Tuning)"
        else:
            self._current_model_type = f"{model_type} (HP Tuning)"
        self._current_training_type = "hyperparameter_tuning"
        
        # Acquire lock to prevent simultaneous training
        self._acquire_lock("hyperparameter_tuning", output_dir)
        
        cmd = [
            sys.executable, "-m", "src.vin_ocr.training.hyperparameter_tuning.optuna_tuning",
            "--model", config.get("model_type", "paddleocr"),
            "--n-trials", str(config.get("n_trials", 50)),
            "--output", output_dir,
        ]
        
        # Add data paths
        if config.get("train_data_dir"):
            cmd.extend(["--train-data", config.get("train_data_dir")])
        if config.get("train_labels"):
            cmd.extend(["--train-labels", config.get("train_labels")])
        if config.get("val_data_dir"):
            cmd.extend(["--val-data", config.get("val_data_dir")])
        if config.get("val_labels"):
            cmd.extend(["--val-labels", config.get("val_labels")])
        
        # Add device
        if config.get("device"):
            cmd.extend(["--device", config.get("device")])
        
        # Add timeout
        if config.get("timeout"):
            cmd.extend(["--timeout", str(config.get("timeout"))])
        
        # Add study name for persistence
        if config.get("study_name"):
            cmd.extend(["--study-name", config.get("study_name")])
        
        # Add storage for persistence (SQLite)
        if config.get("storage"):
            cmd.extend(["--storage", config.get("storage")])
        
        logger.info(f"✓ Starting Hyperparameter Tuning")
        logger.info(f"  Model: {model_type}")
        logger.info(f"  Training Script: src.vin_ocr.training.hyperparameter_tuning.optuna_tuning")
        logger.info(f"  Output Directory: {output_dir}")
        logger.info(f"  Command: {' '.join(cmd)}")
        
        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                cwd=str(Path(__file__).parent.parent.parent.parent),  # Project root
            )
            
            # Start monitoring thread with output_dir and model_tag
            thread = threading.Thread(
                target=self._monitor_tuning_process,
                args=(output_dir, model_type),
                daemon=True
            )
            self._monitor_thread = thread
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
            self._current_training_type = None
            self._release_lock()  # Release lock on error
            raise
    
    def _monitor_tuning_process(self, output_dir: str = "./output/hyperparameter_tuning", model_tag: str = "paddleocr"):
        """Monitor Optuna tuning process output and update tracker."""
        # Capture the process handle in a local: stop() and this thread's own
        # finally block null self._process, and dereferencing the attribute
        # mid-run raced against that (W-M2).
        proc = self._process
        if proc is None:
            return
        
        trial = 0
        best_value = 0.0
        
        # Create console log file for real-time UI display with model-specific name
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        console_log_path = self._build_console_log_path(output_dir, f"{model_tag}_tuning")
        
        # Store the log path for UI to access
        self._current_console_log = console_log_path
        
        try:
            # Clear previous log and write header
            with open(console_log_path, 'w') as console_log:
                start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                console_log.write(f"=" * 60 + "\n")
                console_log.write(f"Hyperparameter Tuning Started: {start_time}\n")
                console_log.write(f"Model: {model_tag}\n")
                console_log.write(f"Output Directory: {output_dir}\n")
                console_log.write(f"=" * 60 + "\n\n")
                console_log.flush()
            
            # Use append mode for real-time writing
            with open(console_log_path, 'a') as console_log:
                for line in proc.stdout:
                    if self._stop_requested:
                        console_log.write(f"\n[STOPPED] Tuning stopped by user\n")
                        console_log.flush()
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Write to console log file IMMEDIATELY for real-time display
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
                    console_log.write(f"[{timestamp}] {line}\n")
                    console_log.flush()
                    
                    # Also write to os sync for immediate disk write
                    import os
                    os.fsync(console_log.fileno())
                    
                    # Parse Optuna output for progress
                    # Expected formats:
                    # "Trial 5 finished with value: 0.85"
                    # "[I 2026-02-01] Trial 5 finished with value: 0.85"
                    
                    import re
                    
                    # Extract trial number
                    trial_match = re.search(r'Trial\s+(\d+)', line)
                    if trial_match:
                        trial = int(trial_match.group(1))
                    
                    # Extract best value
                    value_match = re.search(r'value:\s*(\d+\.?\d*)', line)
                    if value_match:
                        value = float(value_match.group(1))
                        if value > best_value:
                            best_value = value
                    
                    # Update tracker (use trial as epoch, best_value as accuracy)
                    self._tracker.update(trial, 0, 0.0, best_value, line[:100])
                
                # Write completion status
                return_code = proc.wait()
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                console_log.write(f"\n" + "=" * 60 + "\n")
                console_log.write(f"Tuning Ended: {end_time}\n")
                console_log.write(f"Exit Code: {return_code}\n")
                console_log.write(f"Best Value: {best_value:.4f}\n")
                console_log.write(f"=" * 60 + "\n")
                console_log.flush()
            
            # User-stop wins over this thread's completion/error paths:
            # stop() sets the terminal "stopped" status itself (W-M2/W-M11).
            if self._stop_requested:
                pass
            elif return_code == 0:
                self._tracker.complete(f"Hyperparameter tuning completed. Best value: {best_value:.4f}")
            else:
                self._tracker.error(f"Tuning failed with code {return_code}")
                
        except Exception as e:
            logger.exception(f"Error monitoring tuning: {e}")
            if not self._stop_requested:
                self._tracker.error(str(e))
        finally:
            self._process = None
            self._current_model_type = None  # Clear on completion
            self._current_training_type = None
            self._release_lock()  # Always release lock when tuning ends
    
    @staticmethod
    def _pid_alive(pid: int) -> bool:
        """True when a process with this PID exists (signal 0 probe)."""
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists but owned by another user
        except OSError:
            return False
    
    def _read_lock_pid(self) -> Optional[int]:
        """PID recorded in the lock file, or None when absent/unreadable."""
        if not self._lock_file.exists():
            return None
        try:
            with open(self._lock_file, 'r') as f:
                lock_data = json.load(f)
            pid = lock_data.get('pid')
            return int(pid) if pid is not None else None
        except (json.JSONDecodeError, OSError, TypeError, ValueError) as e:
            logger.debug(f"Could not read training lock {self._lock_file}: {e}")
            return None
    
    @staticmethod
    def _terminate_child(proc: subprocess.Popen) -> Tuple[bool, bool]:
        """
        Terminate a child process: SIGTERM, then SIGKILL after 30s.
        
        Returns:
            (exited, graceful): exited=False means the process survived even
            SIGKILL; graceful=True means it exited on SIGTERM.
        """
        logger.info("Sending SIGTERM for graceful shutdown...")
        proc.terminate()
        try:
            # Give the process time to run its shutdown path (up to 30s)
            proc.wait(timeout=30)
            logger.info("Training process terminated gracefully")
            return True, True
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timeout, sending SIGKILL...")
            proc.kill()
            try:
                proc.wait(timeout=5)
                return True, False
            except subprocess.TimeoutExpired:
                logger.error("Training process did not exit after SIGKILL")
                return False, False
    
    def _terminate_locked_pid(self) -> Tuple[bool, bool]:
        """
        Terminate the training process recorded in the lock file (W-M3).
        
        Covers the server-restart case: is_running derives from the lock file
        of a process this runner never started (self._process is None).
        
        Returns:
            (exited, graceful): exited=True when no live process remains
            (including "nothing was running"); graceful=True when it exited
            on SIGTERM (or nothing had to be signalled).
        """
        pid = self._read_lock_pid()
        if pid is None or pid == os.getpid() or not self._pid_alive(pid):
            return True, True  # nothing (left) to stop
        
        logger.info(f"Stopping training process from lock file (PID {pid})...")
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            logger.error(f"Could not send SIGTERM to PID {pid}: {e}")
            return (not self._pid_alive(pid)), False
        
        deadline = time.time() + 30
        while time.time() < deadline:
            if not self._pid_alive(pid):
                return True, True
            time.sleep(0.5)
        
        logger.warning(f"PID {pid} ignored SIGTERM, sending SIGKILL...")
        try:
            os.kill(pid, getattr(signal, "SIGKILL", signal.SIGTERM))
        except OSError as e:
            logger.error(f"Could not send SIGKILL to PID {pid}: {e}")
            return (not self._pid_alive(pid)), False
        
        deadline = time.time() + 5
        while time.time() < deadline:
            if not self._pid_alive(pid):
                return True, False
            time.sleep(0.2)
        return False, False
    
    @staticmethod
    def _stopped_message(graceful: bool, training_type: Optional[str]) -> str:
        """Honest per-trainer stop message (W-M11)."""
        if not graceful:
            return ("Training stopped by user (process had to be force-killed; "
                    "no shutdown checkpoint was written)")
        if training_type == "paddleocr_finetune":
            # finetune_paddleocr registers a SIGTERM handler and saves a
            # `latest` resume checkpoint during graceful shutdown (verified:
            # its train loop calls save_checkpoint on the shutdown flag).
            return ("Training stopped by user (the fine-tune trainer saves a "
                    "`latest` resume checkpoint on SIGTERM)")
        # finetune_deepseek / train_from_scratch / optuna_tuning install no
        # SIGTERM handler: the process just exits. Only checkpoints already
        # written during training remain — claim nothing more.
        return "Training stopped by user (no shutdown checkpoint is saved by this trainer)"
    
    def stop(self) -> bool:
        """
        Stop the running training process with graceful shutdown.
        
        Handles both a child started by this runner and — after a server
        restart — a process known only from the lock file (W-M3). The lock
        file is only removed once its process is confirmed dead; a process
        that survives SIGKILL is reported honestly and the lock retained.
        
        Returns:
            True when nothing is left running (status set to "stopped"),
            False when the process could not be terminated (status set to
            an error, lock left in place).
        """
        self._stop_requested = True
        # Local captures: the monitor thread's finally block nulls
        # self._process and self._current_training_type; dereferencing the
        # attributes mid-stop raced against that (W-M2).
        proc = self._process
        monitor = self._monitor_thread
        training_type = self._current_training_type
        
        if proc is not None and proc.poll() is None:
            exited, graceful = self._terminate_child(proc)
        elif proc is None:
            exited, graceful = self._terminate_locked_pid()
        else:
            exited, graceful = True, True  # child already exited on its own
        
        if not exited:
            # Never delete the lock while its process is alive: that would
            # let a second training start against a live one.
            self._tracker.error(
                "Stop failed: the training process is still running and could "
                "not be terminated; the training lock was left in place."
            )
            return False
        
        # Let the monitor thread finish BEFORE the final status is set: its
        # completion/error paths are guarded by _stop_requested, so after the
        # join the "stopped" status below cannot be overwritten (W-M2).
        if monitor is not None and monitor is not threading.current_thread() and monitor.is_alive():
            monitor.join(timeout=15)
        
        message = self._stopped_message(graceful, training_type)
        self._process = None
        self._monitor_thread = None
        self._current_model_type = None  # Clear model type
        self._current_console_log = None  # Clear console log path
        self._current_training_type = None
        self._release_lock()  # Release lock when stopping
        self._tracker.mark_stopped(message)
        return True
    
    def _monitor_process(self, output_dir: str = "./output/vin_rec_finetune", model_tag: str = "paddleocr"):
        """Monitor training process output and update tracker with real-time console logging."""
        # Capture the process handle in a local: stop() and this thread's own
        # finally block null self._process, and dereferencing the attribute
        # mid-run raced against that (W-M2).
        proc = self._process
        if proc is None:
            return
        
        epoch = 1
        batch = 0
        last_update_time = time.time()
        lines_since_update = 0
        
        import re
        
        # Create console log file for real-time UI display with model-specific name
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        console_log_path = self._build_console_log_path(output_dir, model_tag)
        
        # Store the log path for UI to access
        self._current_console_log = console_log_path
        
        try:
            # Clear previous log and write header
            with open(console_log_path, 'w') as console_log:
                start_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                console_log.write(f"=" * 60 + "\n")
                console_log.write(f"Training Started: {start_time}\n")
                console_log.write(f"Output Directory: {output_dir}\n")
                console_log.write(f"=" * 60 + "\n\n")
                console_log.flush()
            
            # Use append mode for real-time writing
            with open(console_log_path, 'a') as console_log:
                for line in proc.stdout:
                    if self._stop_requested:
                        console_log.write(f"\n[STOPPED] Training stopped by user\n")
                        console_log.flush()
                        break
                    
                    line = line.strip()
                    if not line:
                        continue
                    
                    lines_since_update += 1
                    
                    # Write to console log file IMMEDIATELY for real-time display
                    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]  # Include milliseconds
                    console_log.write(f"[{timestamp}] {line}\n")
                    console_log.flush()  # Critical: flush immediately for real-time reading
                    
                    # Also write to os sync for immediate disk write
                    import os
                    os.fsync(console_log.fileno())
                    
                    # Always log the raw output for debugging
                    logger.debug(f"Training output: {line}")
                    
                    # Parse training output for progress
                    loss = 0.0
                    accuracy = 0.0
                    
                    # Try to extract epoch
                    if "Epoch" in line or "epoch" in line:
                        epoch_match = re.search(r'[Ee]poch\s*\[?(\d+)', line)
                        if epoch_match:
                            epoch = int(epoch_match.group(1))
                    
                    # Try to extract batch/step
                    if "Batch" in line or "batch" in line or "Step" in line or "step" in line:
                        batch_match = re.search(r'[Bb]atch\s*\[?(\d+)|[Ss]tep\s*\[?(\d+)', line)
                        if batch_match:
                            batch = int(batch_match.group(1) or batch_match.group(2))
                    
                    # Try to extract loss
                    if "loss" in line.lower():
                        loss_match = re.search(r'[Ll]oss[:\s=]+(\d+\.?\d*)', line)
                        if loss_match:
                            loss = float(loss_match.group(1))
                    
                    # Try to extract accuracy
                    if "acc" in line.lower():
                        acc_match = re.search(r'[Aa]cc[uracy]*[:\s=]+(\d+\.?\d*)', line)
                        if acc_match:
                            accuracy = float(acc_match.group(1))
                            if accuracy > 1:
                                accuracy /= 100  # Convert percentage
                    
                    # Update tracker more frequently for real-time feel
                    current_time = time.time()
                    # Update every 1 second or every 3 lines (more frequent)
                    if current_time - last_update_time >= 1.0 or lines_since_update >= 3:
                        self._tracker.update(epoch, batch, loss, accuracy, line[:200])
                        last_update_time = current_time
                        lines_since_update = 0
                
                # Write completion status
                return_code = proc.wait()
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                console_log.write(f"\n" + "=" * 60 + "\n")
                console_log.write(f"Training Ended: {end_time}\n")
                console_log.write(f"Exit Code: {return_code}\n")
                console_log.write(f"=" * 60 + "\n")
                console_log.flush()
            
            # User-stop wins over this thread's completion/error paths: the
            # nonzero exit code after a stop is the SIGTERM/SIGKILL stop()
            # itself sent, and stop() sets the terminal status (W-M2/W-M11).
            if self._stop_requested:
                pass
            elif return_code == 0:
                self._tracker.complete("Training completed successfully")
            else:
                state = self._tracker.get_state()
                last_msg = state.history[-1].message if state.history else "Unknown error"
                self._tracker.error(f"Training failed (code {return_code}): {last_msg}")
                
        except Exception as e:
            logger.exception(f"Error monitoring training: {e}")
            if not self._stop_requested:
                self._tracker.error(str(e))
        finally:
            self._process = None
            self._current_model_type = None  # Clear model type
            self._current_training_type = None
            self._release_lock()  # Always release lock when training ends
    
    def get_console_log_path(self) -> Optional[str]:
        """Get the path to the current console log file."""
        return self._current_console_log


class TrainingUI:
    """
    Streamlit UI components for training management.
    
    This class is primarily for organizing UI rendering functions.
    Most rendering is done directly in app.py.
    """
    
    def __init__(self):
        self.tracker = get_global_tracker()
        self.runner = get_global_runner()
    
    def is_training_active(self) -> bool:
        """Check if training is active."""
        return self.runner.is_running
    
    def get_progress(self) -> float:
        """Get training progress as 0-1 float."""
        state = self.tracker.get_state()
        if state.total_epochs == 0:
            return 0.0
        
        total = state.total_epochs * state.total_batches
        current = (state.current_epoch - 1) * state.total_batches + state.current_batch
        return current / total if total > 0 else 0.0
    
    @staticmethod
    def get_hardware_info() -> Dict[str, Any]:
        """Get hardware detection information for display in UI."""
        try:
            # Use absolute import to avoid relative import issues
            from src.vin_ocr.utils.hardware_utils import HardwareDetector
            detector = HardwareDetector()
            info = detector.detect()
            
            # Convert HardwareInfo dataclass to dict for UI
            gpu_devices = []
            for gpu in info.gpus:
                gpu_devices.append({
                    'name': gpu.name,
                    'memory_gb': gpu.total_memory_gb,
                    'type': gpu.device_type.value,
                })
            
            return {
                'platform': info.platform,
                'python_version': info.python_version,
                'cpu_cores': info.cpu_count,
                'cpu_name': info.cpu_name,
                'gpu': {
                    'available': len(info.gpus) > 0,
                    'devices': gpu_devices,
                    'total_memory_gb': info.total_gpu_memory_gb,
                    'device_type': info.device_type.value,
                },
                'cuda': {
                    'available': info.cuda_available,
                    'version': info.cuda_version,
                },
                'mps': {
                    'available': info.mps_available,
                },
                'libraries': {
                    'torch': info.torch_available,
                    'torch_version': info.torch_version,
                    'paddle': info.paddle_available,
                    'paddle_version': info.paddle_version,
                    'bitsandbytes': info.bitsandbytes_available,
                    'peft': True,  # Assume PEFT available if this code runs
                },
                'quantization_supported': info.quantization_supported,
            }
        except ImportError as e:
            return {"error": f"Hardware detection utility not available: {e}"}
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def get_training_recommendations(training_type: str = "paddleocr") -> Dict[str, Any]:
        """Get hardware-based training recommendations."""
        try:
            # Use absolute import to avoid relative import issues
            from src.vin_ocr.utils.hardware_utils import HardwareDetector
            detector = HardwareDetector()
            return detector.get_training_config(training_type)
        except ImportError as e:
            return {"error": f"Hardware detection utility not available: {e}"}
        except Exception as e:
            return {"error": str(e)}


# Global singleton accessors
_global_tracker: Optional[ProgressTracker] = None
_global_runner: Optional[TrainingRunner] = None


def get_global_tracker() -> ProgressTracker:
    """Get the global progress tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = ProgressTracker()
    return _global_tracker


def get_global_runner() -> TrainingRunner:
    """Get the global training runner instance."""
    global _global_runner
    if _global_runner is None:
        _global_runner = TrainingRunner()
    return _global_runner


def reset_global_state():
    """
    Reset global singleton state for testing.
    
    This function clears all global state, making it possible to
    run isolated unit tests. Call this in test setUp/tearDown.
    """
    global _global_tracker, _global_runner
    
    # Reset tracker
    if _global_tracker is not None:
        _global_tracker.reset()
    _global_tracker = None
    
    # Stop any running training and reset runner
    if _global_runner is not None:
        if _global_runner.is_running:
            _global_runner.stop()
        _global_runner._release_lock()
    _global_runner = None
    
    # Also reset singleton class instances
    ProgressTracker._instance = None
    TrainingRunner._instance = None


__all__ = [
    "TrainingState",
    "TrainingUpdate",
    "ProgressTracker",
    "TrainingRunner",
    "TrainingUI",
    "get_global_tracker",
    "get_global_runner",
    "reset_global_state",
    "build_finetune_config",
    "write_finetune_config",
    "DEFAULT_FINETUNE_BASE_CONFIG",
    "FINETUNE_ALGORITHM_MAP",
]
