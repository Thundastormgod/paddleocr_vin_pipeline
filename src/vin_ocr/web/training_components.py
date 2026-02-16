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
from typing import Dict, List, Optional, Any, Callable
import logging

logger = logging.getLogger(__name__)

# Project root for resolving relative paths
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


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
    _lock_file = Path("./output/.training_lock")
    
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
            self._current_console_log = None  # Track current console log path
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
        
        # Validate device selection
        device = config.get("device", "").lower()
        valid_devices = ["cpu", "cuda", "mps", "gpu", "gpu (cuda)", "gpu (cuda - paddle)", "mps (apple silicon)"]
        if device and not any(d in device.lower() for d in ["cpu", "cuda", "mps", "gpu"]):
            errors.append(f"Invalid device: {device}")
        
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
                import json
                with open(self._lock_file, 'r') as f:
                    lock_data = json.load(f)
                pid = lock_data.get('pid')
                # Check if process is still running
                if pid:
                    import os
                    try:
                        os.kill(pid, 0)  # Doesn't kill, just checks if exists
                        return True
                    except OSError:
                        # Process not running, clean up stale lock
                        self._lock_file.unlink()
            except:
                pass
        
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
        """Start PaddleOCR fine-tuning process."""
        if self.is_running:
            raise RuntimeError("Training already in progress. Please wait for it to complete or stop it first.")
        
        # Validate configuration before starting
        self._validate_training_config(config, "paddleocr_finetune")
        
        self._tracker.start(config.get("epochs", 10))
        self._stop_requested = False
        
        # Create timestamped, model-specific output directory
        base_output_dir = config.get("output_dir", "./output/vin_rec_finetune")
        output_dir = self._build_output_dir(base_output_dir, "paddleocr_finetune")
        
        # Set current model type for UI - include architecture if specified
        architecture = config.get("architecture", "")
        if architecture:
            # Use clean architecture name for display
            if "PP-OCRv5" in architecture or "PP_OCRv5" in architecture:
                self._current_model_type = "PP-OCRv5"
            elif "PP-OCRv4" in architecture or "PP_OCRv4" in architecture:
                self._current_model_type = "PP-OCRv4"
            elif "PP-OCRv3" in architecture or "PP_OCRv3" in architecture:
                self._current_model_type = "PP-OCRv3"
            elif "SVTR_LCNet" in architecture:
                self._current_model_type = "SVTR_LCNet"
            elif "SVTR_Tiny" in architecture:
                self._current_model_type = "SVTR_Tiny"
            elif "CRNN" in architecture:
                self._current_model_type = "CRNN"
            else:
                self._current_model_type = architecture
        else:
            self._current_model_type = "PaddleOCR"
        
        # Acquire lock to prevent simultaneous training
        self._acquire_lock("paddleocr_finetune", output_dir)
        
        cmd = [
            sys.executable, "-m", "src.vin_ocr.training.finetune_paddleocr",
            "--epochs", str(config.get("epochs", 10)),
            "--batch-size", str(config.get("batch_size", 8)),
            "--lr", str(config.get("learning_rate", 0.0005)),
            "--output-dir", output_dir,
        ]
        
        # Device selection - explicitly pass GPU or CPU flag
        device = config.get("device", "cpu")
        if device == "cpu":
            cmd.append("--cpu")
        else:
            cmd.append("--gpu")
        
        logger.info(f"  Device: {device.upper()}")
        
        # Add architecture if specified (PP-OCRv5, SVTR_LCNet, CRNN)
        architecture = config.get("architecture", "")
        if architecture:
            # Extract architecture name from display format
            if "PP-OCRv5" in architecture or "PP_OCRv5" in architecture:
                cmd.extend(["--architecture", "PP-OCRv5"])
            elif "SVTR_LCNet" in architecture:
                cmd.extend(["--architecture", "SVTR_LCNet"])
            elif "SVTR_Tiny" in architecture:
                cmd.extend(["--architecture", "SVTR_Tiny"])
            elif "CRNN" in architecture:
                cmd.extend(["--architecture", "CRNN"])
        
        # Add data paths if provided
        if config.get("train_data_dir"):
            cmd.extend(["--train-data-dir", config.get("train_data_dir")])
        if config.get("train_labels"):
            cmd.extend(["--train-labels", config.get("train_labels")])
        if config.get("val_data_dir"):
            cmd.extend(["--val-data-dir", config.get("val_data_dir")])
        if config.get("val_labels"):
            cmd.extend(["--val-labels", config.get("val_labels")])
        
        logger.info(f"✓ Starting PaddleOCR Fine-Tuning")
        logger.info(f"  Model: PaddleOCR")
        logger.info(f"  Training Script: src.vin_ocr.training.finetune_paddleocr")
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
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
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
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
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
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
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
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
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
            thread.start()
            
        except Exception as e:
            self._tracker.error(str(e))
            self._current_model_type = None  # Clear on error
            self._release_lock()  # Release lock on error
            raise
    
    def _monitor_tuning_process(self, output_dir: str = "./output/hyperparameter_tuning", model_tag: str = "paddleocr"):
        """Monitor Optuna tuning process output and update tracker."""
        if self._process is None:
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
                for line in self._process.stdout:
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
                return_code = self._process.wait()
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                console_log.write(f"\n" + "=" * 60 + "\n")
                console_log.write(f"Tuning Ended: {end_time}\n")
                console_log.write(f"Exit Code: {return_code}\n")
                console_log.write(f"Best Value: {best_value:.4f}\n")
                console_log.write(f"=" * 60 + "\n")
                console_log.flush()
            
            if return_code == 0:
                self._tracker.complete(f"Hyperparameter tuning completed. Best value: {best_value:.4f}")
            else:
                self._tracker.error(f"Tuning failed with code {return_code}")
                
        except Exception as e:
            logger.exception(f"Error monitoring tuning: {e}")
            self._tracker.error(str(e))
        finally:
            self._process = None
            self._current_model_type = None  # Clear on completion
            self._release_lock()  # Always release lock when tuning ends
    
    def stop(self):
        """Stop the running training process with graceful shutdown."""
        self._stop_requested = True
        if self._process is not None:
            # First, try graceful shutdown with SIGTERM
            # The training script handles SIGTERM and saves checkpoint
            logger.info("Sending SIGTERM for graceful shutdown...")
            self._process.terminate()
            try:
                # Give process time to save checkpoint (up to 30 seconds)
                self._process.wait(timeout=30)
                logger.info("Training process terminated gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown takes too long
                logger.warning("Graceful shutdown timeout, sending SIGKILL...")
                self._process.kill()
                self._process.wait(timeout=5)
            self._process = None
        self._current_model_type = None  # Clear model type
        self._current_console_log = None  # Clear console log path
        self._release_lock()  # Release lock when stopping
        self._tracker.complete("Training stopped by user (checkpoint saved)")
    
    def _monitor_process(self, output_dir: str = "./output/vin_rec_finetune", model_tag: str = "paddleocr"):
        """Monitor training process output and update tracker with real-time console logging."""
        if self._process is None:
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
                for line in self._process.stdout:
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
                return_code = self._process.wait()
                end_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                console_log.write(f"\n" + "=" * 60 + "\n")
                console_log.write(f"Training Ended: {end_time}\n")
                console_log.write(f"Exit Code: {return_code}\n")
                console_log.write(f"=" * 60 + "\n")
                console_log.flush()
            
            if return_code == 0:
                self._tracker.complete("Training completed successfully")
            else:
                state = self._tracker.get_state()
                last_msg = state.history[-1].message if state.history else "Unknown error"
                self._tracker.error(f"Training failed (code {return_code}): {last_msg}")
                
        except Exception as e:
            logger.exception(f"Error monitoring training: {e}")
            self._tracker.error(str(e))
        finally:
            self._process = None
            self._current_model_type = None  # Clear model type
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
]
