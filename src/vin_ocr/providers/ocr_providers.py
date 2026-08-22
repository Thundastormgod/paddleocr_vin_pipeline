"""
OCR Providers - Multi-Model Abstraction Layer
==============================================

Provides a unified interface for multiple OCR backends:
- PaddleOCR (default, local)
- DeepSeek Vision (API-based)
- Future: TesseractOCR, Google Vision, Azure Vision, etc.

Usage:
    from ocr_providers import OCRProviderFactory, OCRProviderType
    
    # Create a provider
    provider = OCRProviderFactory.create(OCRProviderType.PADDLEOCR)
    result = provider.recognize(image)
    
    # Or use DeepSeek-OCR (local model, no API key)
    provider = OCRProviderFactory.create(OCRProviderType.DEEPSEEK)
    result = provider.recognize(image)

Author: JRL-VIN Project
Date: January 2026
"""

import os
import base64
import dataclasses
import time
import logging
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import cv2

# Import VIN constants and the checksum-scored extractor (Single Source of Truth)
from ..core.vin_utils import VINConstants, extract_vin_from_text
from config import get_config

# Import VIN preprocessing module
from ..preprocessing import VINPreprocessor, PreprocessConfig, PreprocessStrategy

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND DATA CLASSES
# =============================================================================

class OCRProviderType(str, Enum):
    """
    Supported OCR provider types.
    
    Only includes providers that are FULLY IMPLEMENTED.
    Use OCRProviderFactory.list_available() to check availability.
    """
    PADDLEOCR = "paddleocr"
    DEEPSEEK = "deepseek"
    # Note: Additional providers (Tesseract, Google Vision, Azure, OpenAI) 
    # can be added by subclassing OCRProvider and registering with the factory.


@dataclass
class OCRResult:
    """
    Standardized OCR result across all providers.
    
    Attributes:
        text: Recognized text string
        confidence: Confidence score (0.0 to 1.0)
        raw_response: Provider-specific raw response for debugging
        bounding_boxes: List of detected text regions (optional)
        provider: Name of the OCR provider used
        metadata: Additional provider-specific metadata
    """
    text: str
    confidence: float
    raw_response: Any = None
    bounding_boxes: List[Dict[str, Any]] = field(default_factory=list)
    provider: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "text": self.text,
            "confidence": self.confidence,
            "provider": self.provider,
            "bounding_boxes": self.bounding_boxes,
            "metadata": self.metadata,
        }


@dataclass
class ProviderConfig:
    """Base configuration for OCR providers."""
    timeout: int = 30  # seconds
    max_retries: int = 3
    retry_delay: float = 1.0  # seconds
    
    # Preprocessing configuration (shared by all providers)
    preprocess_enabled: bool = True
    preprocess_strategy: PreprocessStrategy = PreprocessStrategy.ENGRAVED
    preprocess_target_width: int = 1024
    preprocess_clahe_clip_limit: float = 2.0


@dataclass
class PaddleOCRConfig(ProviderConfig):
    """
    PaddleOCR-specific configuration.

    Field names map onto the PaddleOCR 3.x constructor except where noted.
    """
    lang: str = "en"
    use_gpu: bool = True
    det_db_box_thresh: float = 0.3
    # DEPRECATED / INERT: PaddleOCR 3.x removed `rec_thresh` (passing it raises
    # "ValueError: Unknown argument"). It is retained only so existing configs
    # and callers keep working, and is deliberately NOT forwarded to
    # PaddleOCR() in initialize(). Filter low-confidence results downstream
    # instead. Scheduled for removal in 2.0.
    rec_thresh: float = 0.3
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    ocr_version: str = "PP-OCRv3"  # PP-OCRv3 works better for VIN plates


@dataclass
class DeepSeekOCRConfig(ProviderConfig):
    """
    DeepSeek-OCR local model configuration.
    
    DeepSeek-OCR is a Vision-Language Model for OCR from DeepSeek AI:
    - GitHub: https://github.com/deepseek-ai/DeepSeek-OCR
    - HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-OCR
    - Paper: https://arxiv.org/abs/2510.18234
    
    Backends:
    - pytorch: Standard HuggingFace transformers with model.infer() API
    - vllm: Production-grade batched inference (recommended for throughput)
    
    Resolution Presets (from official docs):
    - Tiny: base_size=512, image_size=512, crop_mode=False (64 vision tokens)
    - Small: base_size=640, image_size=640, crop_mode=False (100 vision tokens)
    - Base: base_size=1024, image_size=1024, crop_mode=False (256 vision tokens)
    - Large: base_size=1280, image_size=1280, crop_mode=False (400 vision tokens)
    - Gundam: base_size=1024, image_size=640, crop_mode=True (dynamic, best for VIN)
    
    Requirements:
    - transformers>=4.51.1 (or vllm>=0.8.5)
    - torch>=2.6.0 with CUDA 11.8+
    - flash-attn>=2.7.3 (optional, for speed)
    """
    # Model identification - official DeepSeek-OCR model
    model_name: str = "deepseek-ai/DeepSeek-OCR"
    finetuned_model_path: Optional[str] = None  # Local full fine-tuned model path
    adapter_path: Optional[str] = None  # PEFT adapter path (LoRA/QLoRA)
    merge_adapter: bool = False  # Merge adapter weights into base model
    
    # Model caching for faster subsequent loads
    cache_dir: Optional[str] = None  # Defaults to ~/.cache/huggingface
    
    # Hardware configuration
    use_gpu: bool = True
    use_flash_attention: bool = True
    device: Optional[str] = None  # Auto-detect if None: "cuda", "mps", or "cpu"
    
    # Backend selection
    backend: str = "pytorch"  # Options: "pytorch", "vllm"
    
    # Resolution settings (Gundam preset - best for VIN plates)
    base_size: int = 1024
    image_size: int = 640
    crop_mode: bool = True  # Enable dynamic resolution for better small text (VINs)
    
    # Generation settings
    max_tokens: int = 8192  # DeepSeek-OCR supports up to 8192 tokens
    
    # VIN-optimized prompt (Free OCR mode for clean text extraction)
    prompt: str = "<image>\nFree OCR."
    
    # vLLM backend settings
    use_vllm: bool = False  # Set True for production throughput
    vllm_ngram_size: int = 30
    vllm_window_size: int = 90
    
    # Output settings (for transformers backend)
    save_results: bool = False
    test_compress: bool = False
    output_path: Optional[str] = None
    
    def __post_init__(self):
        """Auto-detect best device if not specified."""
        # Keep `backend` and the legacy `use_vllm` flag coherent in BOTH
        # directions. Historically only use_vllm -> backend was mapped, so
        # DeepSeekOCRConfig(backend="vllm") loaded vLLM in initialize() but
        # recognize() still routed to the (never-loaded) transformers model
        # and crashed with AttributeError.
        if self.use_vllm and self.backend == "pytorch":
            self.backend = "vllm"
        if self.backend == "vllm":
            self.use_vllm = True
        
        # Set default cache directory
        if self.cache_dir is None:
            self.cache_dir = str(Path.home() / ".cache" / "huggingface")
        
        if self.device is None:
            try:
                import torch
                if torch.cuda.is_available():
                    self.device = "cuda"
                elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                    self.device = "mps"
                else:
                    self.device = "cpu"
            except ImportError:
                self.device = "cpu"


# =============================================================================
# ABSTRACT BASE CLASS
# =============================================================================

class OCRProvider(ABC):
    """
    Abstract base class for OCR providers.
    
    All OCR backends must implement this interface to ensure
    consistent behavior across the pipeline.
    
    Includes built-in preprocessing via VINPreprocessor for consistent
    image enhancement across all providers.
    
    Thread Safety: Implementations should be thread-safe.
    """
    
    # Subclasses should initialize this in __init__
    _initialized: bool = False
    _preprocessor: Optional[VINPreprocessor] = None
    
    def _init_preprocessor(self, config: ProviderConfig) -> None:
        """Initialize the VIN image preprocessor based on config."""
        if config.preprocess_enabled:
            preprocess_config = PreprocessConfig(
                strategy=config.preprocess_strategy,
                target_width=config.preprocess_target_width,
                clahe_clip_limit=config.preprocess_clahe_clip_limit,
            )
            self._preprocessor = VINPreprocessor(config=preprocess_config)
            logger.debug(f"Preprocessor initialized with strategy={config.preprocess_strategy.value}")
        else:
            self._preprocessor = None
            logger.debug("Preprocessing disabled")
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply VIN-optimized preprocessing to image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Preprocessed image (BGR format)
        """
        if self._preprocessor is None:
            return image
        return self._preprocessor.process(image)
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the provider name."""
        ...
    
    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the provider is available and properly configured."""
        ...
    
    @property
    def is_initialized(self) -> bool:
        """Check if the provider has been initialized."""
        return self._initialized
    
    @abstractmethod
    def initialize(self) -> None:
        """
        Initialize the OCR engine.
        
        Raises:
            OCRProviderError: If initialization fails
        """
        ...
    
    @abstractmethod
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        **kwargs
    ) -> OCRResult:
        """
        Recognize text from an image.
        
        Args:
            image: Image path, URL, or numpy array (BGR format)
            **kwargs: Provider-specific options
            
        Returns:
            OCRResult with recognized text and confidence
            
        Raises:
            OCRProviderError: If recognition fails
        """
        ...

    def recognize_with_retry(
        self,
        image: Union[str, Path, np.ndarray],
        **kwargs
    ) -> OCRResult:
        """
        Recognize text with retry/backoff using provider config.
        """
        config = getattr(self, "config", None)
        max_retries = getattr(config, "max_retries", 1) or 1
        retry_delay = getattr(config, "retry_delay", 0.0) or 0.0

        def _should_retry(exc: Exception) -> bool:
            return isinstance(exc, (OCRProviderError, TimeoutError, ConnectionError))

        last_exception: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                return self.recognize(image, **kwargs)
            except Exception as exc:
                last_exception = exc
                if attempt >= max_retries - 1 or not _should_retry(exc):
                    raise
                sleep_for = retry_delay * (2 ** attempt)
                if sleep_for > 0:
                    logger.warning(
                        "Retrying OCR provider %s after error (attempt %d/%d, sleep %.2fs): %s",
                        self.name,
                        attempt + 1,
                        max_retries,
                        sleep_for,
                        exc,
                    )
                    time.sleep(sleep_for)

        if last_exception:
            raise last_exception
        raise OCRProviderError("Recognition failed without exception", provider=self.name)
    
    def recognize_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        **kwargs
    ) -> List[OCRResult]:
        """
        Recognize text from multiple images.
        
        Default implementation processes images sequentially.
        Providers may override for batch optimization.
        
        Args:
            images: List of image paths or numpy arrays
            **kwargs: Provider-specific options
            
        Returns:
            List of OCRResult objects
        """
        return [self.recognize_with_retry(img, **kwargs) for img in images]
    
    def _load_image(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """
        Load image from path or return numpy array.
        
        Args:
            image: Image path or numpy array
            
        Returns:
            Image as numpy array (BGR format)
            
        Raises:
            ValueError: If image cannot be loaded
        """
        if isinstance(image, np.ndarray):
            return image
        
        path = Path(image)
        if not path.exists():
            raise ValueError(f"Image file not found: {path}")
        
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Failed to load image: {path}")
        
        return img
    
    def _image_to_base64(self, image: np.ndarray) -> str:
        """
        Convert numpy image to base64 string for API calls.
        
        Args:
            image: Image as numpy array (BGR format)
            
        Returns:
            Base64 encoded JPEG string
        """
        _, buffer = cv2.imencode('.jpg', image)
        return base64.b64encode(buffer).decode('utf-8')


class OCRProviderError(Exception):
    """Base exception for OCR provider errors."""
    
    def __init__(
        self,
        message: str,
        provider: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.provider = provider
        self.details = details or {}
        super().__init__(f"[{provider}] {message}")


# =============================================================================
# PADDLEOCR PROVIDER
# =============================================================================

class PaddleOCRProvider(OCRProvider):
    """
    PaddleOCR-based text recognition provider.
    
    Uses PaddleOCR v3.x with PP-OCRv3 models for high-accuracy
    text detection and recognition on VIN plates.
    
    Features:
    - Local processing (no API calls)
    - GPU acceleration support
    - High accuracy on engraved text
    - Built-in VIN-optimized preprocessing
    """
    
    def __init__(self, config: Optional[PaddleOCRConfig] = None):
        """
        Initialize PaddleOCR provider.
        
        Args:
            config: PaddleOCR configuration (uses defaults if None)
        """
        self.config = config or PaddleOCRConfig()
        self._ocr = None
        self._initialized = False
        
        # Initialize preprocessor from base class
        self._init_preprocessor(self.config)
    
    @property
    def name(self) -> str:
        return "PaddleOCR"
    
    @property
    def is_available(self) -> bool:
        """Check if PaddleOCR is installed."""
        try:
            from paddleocr import PaddleOCR
            return True
        except ImportError:
            return False
    
    def initialize(self) -> None:
        """Initialize PaddleOCR engine."""
        if self._initialized:
            return
        
        if not self.is_available:
            raise OCRProviderError(
                "PaddleOCR is not installed. Run: pip install paddleocr",
                provider=self.name
            )
        
        try:
            from paddleocr import PaddleOCR
            
            # Honor config.use_gpu (previously stored and never acted on -
            # logic-audit L16). Same mechanism as VINOCRPipeline: select the
            # paddle device before constructing the engine, downgrading loudly
            # when a GPU is requested but not present.
            try:
                import paddle
                if self.config.use_gpu:
                    if paddle.device.cuda.device_count() > 0:
                        paddle.device.set_device('gpu')
                        logger.info("PaddleOCR provider using GPU")
                    else:
                        paddle.device.set_device('cpu')
                        logger.warning(
                            "use_gpu=True but no CUDA device is present; "
                            "PaddleOCR provider falling back to CPU"
                        )
                else:
                    paddle.device.set_device('cpu')
            except Exception as e:
                logger.warning(f"Could not set paddle device explicitly: {e}")
            
            logger.info(f"Initializing PaddleOCR with {self.config.ocr_version}...")
            self._ocr = PaddleOCR(
                lang=self.config.lang,
                ocr_version=self.config.ocr_version,
                use_doc_orientation_classify=self.config.use_doc_orientation_classify,
                use_doc_unwarping=self.config.use_doc_unwarping,
                use_textline_orientation=self.config.use_textline_orientation,
                text_det_box_thresh=self.config.det_db_box_thresh,
            )
            self._initialized = True
            logger.info(f"PaddleOCR ({self.config.ocr_version}) initialized successfully")
            logger.info(f"Preprocessing: {'enabled' if self._preprocessor else 'disabled'} "
                       f"(strategy={self.config.preprocess_strategy.value})")
            
        except Exception as e:
            raise OCRProviderError(
                f"Failed to initialize PaddleOCR: {e}",
                provider=self.name,
                details={"error": str(e)}
            ) from e
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        preprocess: Optional[bool] = None,
        preprocess_strategy: Optional[PreprocessStrategy] = None,
        **kwargs
    ) -> OCRResult:
        """
        Recognize text using PaddleOCR.
        
        Args:
            image: Image path or numpy array
            preprocess: Override preprocessing (None=use config)
            preprocess_strategy: Override preprocessing strategy
            **kwargs: Additional options (unused)
            
        Returns:
            OCRResult with recognized text
        """
        if not self._initialized:
            self.initialize()
        
        # Load image
        img = self._load_image(image)
        
        # Apply preprocessing
        should_preprocess = preprocess if preprocess is not None else self.config.preprocess_enabled
        if should_preprocess:
            if preprocess_strategy and self._preprocessor:
                img = self._preprocessor.process(img, strategy=preprocess_strategy)
            else:
                img = self._preprocess_image(img)
        
        # Run OCR
        try:
            result = self._ocr.predict(img)
        except Exception as e:
            raise OCRProviderError(
                f"OCR prediction failed: {e}",
                provider=self.name,
                details={"error": str(e)}
            ) from e
        
        # Extract text and confidence
        text, confidence, boxes = self._parse_result(result)
        
        return OCRResult(
            text=text,
            confidence=confidence,
            raw_response=result,
            bounding_boxes=boxes,
            provider=self.name,
            metadata={"lang": self.config.lang}
        )
    
    def _parse_result(self, result: Any) -> Tuple[str, float, List[Dict]]:
        """Parse PaddleOCR result format."""
        if not result:
            return "", 0.0, []
        
        # Handle PaddleOCR v3.x format (list of dicts)
        if isinstance(result, list):
            if len(result) == 0:
                return "", 0.0, []
            result = result[0]
        
        if isinstance(result, dict):
            texts = result.get('rec_texts', [])
            scores = result.get('rec_scores', [])
            dt_polys = result.get('dt_polys', [])
            
            if texts:
                full_text = ''.join(texts)
                avg_score = float(np.mean(scores)) if scores else 0.0
                
                # Build bounding boxes
                boxes = []
                for i, poly in enumerate(dt_polys):
                    boxes.append({
                        "text": texts[i] if i < len(texts) else "",
                        "confidence": scores[i] if i < len(scores) else 0.0,
                        "polygon": poly.tolist() if hasattr(poly, 'tolist') else poly
                    })
                
                return full_text, avg_score, boxes
        
        return "", 0.0, []


# =============================================================================
# DEEPSEEK-OCR PROVIDER (Local HuggingFace Model)
# =============================================================================

class DeepSeekOCRProvider(OCRProvider):
    """
    DeepSeek-OCR local model provider with full business logic.
    
    Uses the DeepSeek-OCR model for local OCR processing:
    - GitHub: https://github.com/deepseek-ai/DeepSeek-OCR
    - HuggingFace: https://huggingface.co/deepseek-ai/DeepSeek-OCR
    - Paper: https://arxiv.org/abs/2510.18234
    
    Features:
    - Local processing (no API calls, no API key required)
    - GPU/MPS/CPU support with automatic device selection
    - Flash attention 2.0 for faster inference
    - vLLM backend support for production workloads (~2500 tokens/s on A100)
    - Batch processing for multiple images
    - VIN-specific prompting and post-processing
    - Multiple resolution presets (Tiny/Small/Base/Large/Gundam)
    - Built-in VIN-optimized preprocessing
    
    Requirements:
    - transformers>=4.51.1
    - torch>=2.6.0 (CUDA 11.8+)
    - einops, addict, easydict
    - flash-attn>=2.7.3 (optional, for speed)
    - vllm>=0.8.5 (optional, for production inference)
    
    Usage:
        # Basic usage (transformers backend)
        provider = DeepSeekOCRProvider()
        result = provider.recognize("image.jpg")
        
        # With vLLM backend (faster, recommended for production)
        config = DeepSeekOCRConfig(backend="vllm")
        provider = DeepSeekOCRProvider(config)
        
        # Batch processing
        results = provider.recognize_batch(["img1.jpg", "img2.jpg"])
    """
    
    # VIN-optimized prompt (Free OCR mode for clean text extraction)
    VIN_PROMPT = "<image>\nFree OCR."
    
    def __init__(self, config: Optional[DeepSeekOCRConfig] = None, device: Optional[str] = None):
        """
        Initialize DeepSeek-OCR provider.
        
        Args:
            config: DeepSeek-OCR configuration with resolution and device settings
            device: Override device (e.g., "cuda", "mps", "cpu"). Takes precedence over config.
        """
        self.config = config or DeepSeekOCRConfig()
        
        # Allow device override
        if device is not None:
            self.config.device = device
            self.config.use_gpu = device not in ("cpu",)
        
        self._model = None
        self._tokenizer = None
        self._vllm_model = None
        self._initialized = False
        self._device = None
        
        # Initialize preprocessor from base class
        self._init_preprocessor(self.config)
    
    @property
    def name(self) -> str:
        return "DeepSeek-OCR"
    
    @property
    def is_available(self) -> bool:
        """Check if required dependencies are available."""
        try:
            import torch
            import transformers
            # Check for minimum version
            from packaging import version
            if version.parse(transformers.__version__) < version.parse("4.46.0"):
                logger.warning("transformers version should be >= 4.46.0")
                return False
            return True
        except ImportError:
            return False
    
    @property
    def device_info(self) -> Dict[str, Any]:
        """Get information about the compute device being used."""
        if not self._initialized:
            return {"device": "not_initialized", "dtype": None}
        
        try:
            import torch
            if self._device:
                return {
                    "device": str(self._device),
                    "dtype": str(self._model.dtype) if hasattr(self._model, 'dtype') else "unknown",
                    "gpu_name": torch.cuda.get_device_name(0) if self._device.type == "cuda" else None,
                    "gpu_memory_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2) if self._device.type == "cuda" else None
                }
        except Exception as e:
            # Diagnostics-only path: fall through to the config-derived
            # fallback below, but never silently.
            logger.debug(f"device_info probe failed, using config fallback: {e}")
        return {"device": self.config.device, "dtype": None}
    
    def _check_flash_attention(self) -> bool:
        """Check if flash attention is available."""
        try:
            import flash_attn
            return True
        except ImportError:
            return False
    
    def _get_optimal_attention(self) -> str:
        """Determine the best attention implementation available."""
        if self.config.use_flash_attention and self._check_flash_attention():
            return 'flash_attention_2'
        return 'eager'
    
    def initialize(self) -> None:
        """
        Initialize the DeepSeek-OCR model.
        
        Supports backends:
        - pytorch: Standard HuggingFace transformers (default, recommended)
        - vllm: Production-grade batched inference
        
        NOTE: ONNX backend has been removed - VLMs cannot be properly exported
        to ONNX as they require both vision encoder AND language decoder.
        
        Auto-selects best device (CUDA > MPS > CPU).
        """
        if self._initialized:
            return
        
        if not self.is_available:
            raise OCRProviderError(
                "DeepSeek-OCR requires transformers>=4.46.0 and torch>=2.0. "
                "Run: pip install transformers torch einops addict easydict",
                provider=self.name
            )
        
        # Select backend
        backend = self.config.backend.lower()
        
        if backend == "onnx":
            # ONNX backend removed - fall back to PyTorch with warning
            logger.warning(
                "ONNX backend requested but not supported for VLM models. "
                "VLMs require both vision encoder and language decoder which "
                "cannot be properly exported to ONNX. Using PyTorch backend."
            )
            self._initialize_transformers()
        elif backend == "vllm" or self.config.use_vllm:
            self._initialize_vllm()
        else:
            self._initialize_transformers()
    
    # NOTE: The former _initialize_onnx/_get_onnx_providers/_export_to_onnx
    # methods were removed (logic-audit L21). They were unreachable
    # (initialize() routes backend=="onnx" to transformers with a warning),
    # referenced config fields that do not exist on DeepSeekOCRConfig
    # (onnx_model_path/onnx_optimize/onnx_provider - guaranteed
    # AttributeError if ever wired), and the export fallback both fell back
    # AND raised. VLMs need vision encoder + language decoder and cannot be
    # exported to a single usable ONNX graph here.

    def _initialize_transformers(self) -> None:
        """Initialize using HuggingFace transformers backend."""
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer
            
            model_id = self.config.finetuned_model_path or self.config.model_name
            logger.info(f"Loading DeepSeek-OCR model: {model_id}")
            print(f"Loading DeepSeek-OCR model (this may take a while on first run)...")
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                cache_dir=self.config.cache_dir
            )
            
            # Determine attention implementation
            attn_impl = self._get_optimal_attention()
            if attn_impl != 'flash_attention_2' and self.config.use_flash_attention:
                logger.warning("Flash attention requested but not available, using eager attention")
            
            # Load model with optimized settings
            self._model = AutoModel.from_pretrained(
                model_id,
                _attn_implementation=attn_impl,
                trust_remote_code=True,
                use_safetensors=True,
                cache_dir=self.config.cache_dir,
                torch_dtype=torch.bfloat16 if self.config.device != "cpu" else torch.float32
            )

            if self.config.adapter_path:
                try:
                    from peft import PeftModel
                except ImportError as e:
                    raise OCRProviderError(
                        "PEFT adapter requested but 'peft' is not installed.",
                        provider=self.name,
                        details={"error": str(e), "adapter_path": self.config.adapter_path}
                    ) from e

                self._model = PeftModel.from_pretrained(self._model, self.config.adapter_path)
                if self.config.merge_adapter:
                    self._model = self._model.merge_and_unload()
            
            # Move to appropriate device
            self._device = self._select_device(torch)
            if self._device.type == "cuda":
                self._model = self._model.eval().cuda().to(torch.bfloat16)
                logger.info(f"DeepSeek-OCR loaded on {torch.cuda.get_device_name(0)} (bfloat16)")
            elif self._device.type == "mps":
                self._model = self._model.eval().to("mps")
                logger.info("DeepSeek-OCR loaded on Apple MPS")
            else:
                self._model = self._model.eval()
                logger.info("DeepSeek-OCR loaded on CPU (float32)")
            
            self._initialized = True
            print(f"DeepSeek-OCR model loaded successfully on {self._device}!")
            
        except ImportError as e:
            raise OCRProviderError(
                f"Missing dependency for DeepSeek-OCR: {e}. "
                "Run: pip install transformers torch einops addict easydict",
                provider=self.name,
                details={"error": str(e), "dependency": "transformers"}
            ) from e
        except Exception as e:
            raise OCRProviderError(
                f"Failed to initialize DeepSeek-OCR: {e}",
                provider=self.name,
                details={"error": str(e), "model": self.config.model_name}
            ) from e
    
    def _initialize_vllm(self) -> None:
        """Initialize using vLLM backend for production inference."""
        try:
            from vllm import LLM, SamplingParams
            from vllm.model_executor.models.deepseek_ocr import NGramPerReqLogitsProcessor

            if self.config.adapter_path:
                logger.warning("PEFT adapters are not supported with vLLM backend; ignoring adapter_path")
            
            logger.info(f"Loading DeepSeek-OCR with vLLM backend: {self.config.model_name}")
            print("Loading DeepSeek-OCR with vLLM (optimized for production)...")
            
            self._vllm_model = LLM(
                model=self.config.model_name,
                enable_prefix_caching=False,
                mm_processor_cache_gb=0,
                logits_processors=[NGramPerReqLogitsProcessor]
            )
            
            self._initialized = True
            logger.info("DeepSeek-OCR (vLLM) initialized successfully")
            print("DeepSeek-OCR model loaded with vLLM backend!")
            
        except ImportError as e:
            raise OCRProviderError(
                f"vLLM backend requested but not installed: {e}. "
                "Run: pip install vllm --pre --extra-index-url https://wheels.vllm.ai/nightly",
                provider=self.name,
                details={"error": str(e), "backend": "vllm"}
            ) from e
        except Exception as e:
            raise OCRProviderError(
                f"Failed to initialize vLLM backend: {e}",
                provider=self.name,
                details={"error": str(e)}
            ) from e
    
    def _select_device(self, torch_module) -> Any:
        """
        Select the optimal compute device.

        Returns a torch.device. Annotated as Any because torch is imported
        lazily inside method bodies; a 'torch.device' string annotation made
        any annotation introspection (typing.get_type_hints) raise NameError
        (logic-audit A1).
        """
        if self.config.device == "cuda" and torch_module.cuda.is_available():
            return torch_module.device("cuda")
        elif self.config.device == "mps" and hasattr(torch_module.backends, 'mps') and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        elif self.config.device == "cpu":
            return torch_module.device("cpu")
        
        # Auto-select
        if torch_module.cuda.is_available():
            return torch_module.device("cuda")
        elif hasattr(torch_module.backends, 'mps') and torch_module.backends.mps.is_available():
            return torch_module.device("mps")
        return torch_module.device("cpu")
    
    def _prepare_image(self, image: Union[str, Path, np.ndarray]) -> Tuple[str, bool]:
        """
        Prepare image for inference, handling various input types.
        
        Returns:
            Tuple of (image_path, needs_cleanup)
        """
        if isinstance(image, np.ndarray):
            # Validate array
            if image.size == 0:
                raise ValueError("Empty image array provided")
            if len(image.shape) not in [2, 3]:
                raise ValueError(f"Invalid image shape: {image.shape}")
            
            # Save to temp file
            import tempfile
            import os
            
            fd, temp_path = tempfile.mkstemp(suffix='.jpg')
            os.close(fd)
            
            # Ensure BGR format for cv2
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            
            cv2.imwrite(temp_path, image)
            return temp_path, True
        
        # Path-like input
        image_path = str(image)
        if not Path(image_path).exists():
            raise ValueError(f"Image file not found: {image_path}")
        
        return image_path, False
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        prompt: Optional[str] = None,
        preprocess: Optional[bool] = None,
        preprocess_strategy: Optional[PreprocessStrategy] = None,
        **kwargs
    ) -> OCRResult:
        """
        Recognize text using DeepSeek-OCR local model.
        
        Args:
            image: Image path or numpy array (BGR format)
            prompt: Custom prompt (uses VIN-optimized prompt if None)
            preprocess: Override preprocessing (None=use config)
            preprocess_strategy: Override preprocessing strategy
            **kwargs: Additional options:
                - base_size: Override config base_size
                - image_size: Override config image_size
                - crop_mode: Override config crop_mode
            
        Returns:
            OCRResult with recognized text, confidence, and metadata
        """
        if not self._initialized:
            self.initialize()
        
        # Preprocess when enabled - for BOTH ndarray and path inputs. Path
        # inputs previously bypassed preprocessing entirely, making
        # preprocess_enabled silently inert for the most common call form
        # (logic-audit L18). Paths are only loaded/re-encoded when
        # preprocessing actually runs; otherwise they pass through untouched.
        should_preprocess = preprocess if preprocess is not None else self.config.preprocess_enabled
        if should_preprocess and self._preprocessor is not None:
            img = image if isinstance(image, np.ndarray) else self._load_image(image)
            if preprocess_strategy:
                img = self._preprocessor.process(img, strategy=preprocess_strategy)
            else:
                img = self._preprocess_image(img)
            image = img
        elif should_preprocess and self._preprocessor is None:
            logger.debug(
                "Preprocess requested but preprocessor is disabled in config; "
                "image passed through unchanged"
            )
        
        # Prepare image
        image_path, needs_cleanup = self._prepare_image(image)
        
        try:
            # Route on which backend actually initialized, not on config flags:
            # the flags select the backend at initialize() time, but the loaded
            # model is the ground truth for dispatch (see H2 in the logic audit).
            if self._vllm_model is not None:
                return self._recognize_vllm(image_path, prompt, **kwargs)
            else:
                return self._recognize_transformers(image_path, prompt, **kwargs)
        finally:
            # Cleanup temp file
            if needs_cleanup:
                self._cleanup_temp(image_path)
    
    def _recognize_transformers(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Run inference using transformers backend."""
        ocr_prompt = prompt or self.config.prompt or self.VIN_PROMPT
        
        # Get inference parameters (allow per-call overrides)
        base_size = kwargs.get('base_size', self.config.base_size)
        image_size = kwargs.get('image_size', self.config.image_size)
        crop_mode = kwargs.get('crop_mode', self.config.crop_mode)
        
        try:
            # Run model inference
            raw_result = self._model.infer(
                self._tokenizer,
                prompt=ocr_prompt,
                image_file=image_path,
                base_size=base_size,
                image_size=image_size,
                crop_mode=crop_mode,
                save_results=False,
                test_compress=False
            )
            
            # Parse result
            text = self._extract_text(raw_result)
            
            # VIN-specific post-processing
            vin_text = self._extract_vin(text)
            
            # Calculate confidence. The custom model.infer() API returns text
            # only - it exposes no token scores - so this backend can only use
            # the format heuristic. The metadata says so explicitly, so
            # downstream consumers never mistake it for model confidence.
            confidence = self._calculate_confidence(vin_text, text)
            
            return OCRResult(
                text=vin_text if vin_text else text.upper().strip(),
                confidence=confidence,
                raw_response=raw_result,
                provider=self.name,
                metadata={
                    "model": self.config.model_name,
                    "device": str(self._device) if self._device else self.config.device,
                    "base_size": base_size,
                    "image_size": image_size,
                    "crop_mode": crop_mode,
                    "backend": "transformers",
                    "raw_text": text,
                    "confidence_source": "format_heuristic (model.infer exposes no scores)",
                }
            )
            
        except Exception as e:
            raise OCRProviderError(
                f"DeepSeek-OCR inference failed: {e}",
                provider=self.name,
                details={"error": str(e), "image": image_path}
            ) from e
    
    def _recognize_vllm(
        self,
        image_path: str,
        prompt: Optional[str] = None,
        **kwargs
    ) -> OCRResult:
        """Run inference using vLLM backend."""
        from vllm import SamplingParams
        from PIL import Image
        
        ocr_prompt = prompt or self.config.prompt or self.VIN_PROMPT
        
        try:
            # Load image
            pil_image = Image.open(image_path).convert("RGB")
            
            # Prepare input
            model_input = [{
                "prompt": ocr_prompt,
                "multi_modal_data": {"image": pil_image}
            }]
            
            # Configure sampling. logprobs=1 makes vLLM return the chosen
            # token's log-probability at every step, which is the model
            # confidence signal _calculate_confidence needs (H1: without it,
            # every format-valid read got a flat heuristic constant).
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                logprobs=1,
                extra_args={
                    "ngram_size": self.config.vllm_ngram_size,
                    "window_size": self.config.vllm_window_size,
                    "whitelist_token_ids": {128821, 128822},  # <td>, </td>
                },
                skip_special_tokens=False,
            )
            
            # Run inference
            outputs = self._vllm_model.generate(model_input, sampling_params)
            raw_text = outputs[0].outputs[0].text if outputs else ""
            model_conf = self._vllm_sequence_confidence(outputs[0]) if outputs else None
            
            # Post-process
            vin_text = self._extract_vin(raw_text)
            confidence = self._calculate_confidence(
                vin_text, raw_text,
                model_logits={"confidence": model_conf} if model_conf is not None else None,
            )
            
            return OCRResult(
                text=vin_text if vin_text else raw_text.upper().strip(),
                confidence=confidence,
                raw_response=raw_text,
                provider=self.name,
                metadata={
                    "model": self.config.model_name,
                    "backend": "vllm",
                    "raw_text": raw_text,
                    "confidence_source": (
                        "model_logprobs" if model_conf is not None else "format_heuristic"
                    ),
                    "model_confidence": model_conf,
                }
            )
            
        except Exception as e:
            raise OCRProviderError(
                f"DeepSeek-OCR (vLLM) inference failed: {e}",
                provider=self.name,
                details={"error": str(e)}
            ) from e
    
    def recognize_batch(
        self,
        images: List[Union[str, Path, np.ndarray]],
        prompt: Optional[str] = None,
        **kwargs
    ) -> List[OCRResult]:
        """
        Process multiple images efficiently.
        
        With vLLM backend, this uses true batched inference.
        With transformers backend, processes sequentially.
        
        Args:
            images: List of image paths or numpy arrays
            prompt: Custom prompt for all images
            **kwargs: Additional options
            
        Returns:
            List of OCRResult objects
        """
        if not self._initialized:
            self.initialize()
        
        if self._vllm_model is not None:
            return self._batch_recognize_vllm(images, prompt, **kwargs)
        
        # Sequential processing for transformers backend
        results = []
        for image in images:
            try:
                result = self.recognize(image, prompt, **kwargs)
                results.append(result)
            except OCRProviderError as e:
                # Return error result for failed images
                results.append(OCRResult(
                    text="",
                    confidence=0.0,
                    provider=self.name,
                    metadata={"error": str(e)}
                ))
        
        return results
    
    def _batch_recognize_vllm(
        self,
        images: List[Union[str, Path, np.ndarray]],
        prompt: Optional[str] = None,
        **kwargs
    ) -> List[OCRResult]:
        """Batch inference using vLLM."""
        from vllm import SamplingParams
        from PIL import Image
        
        ocr_prompt = prompt or self.config.prompt or self.VIN_PROMPT
        temp_files = []
        
        try:
            # Prepare all images
            model_inputs = []
            for image in images:
                image_path, needs_cleanup = self._prepare_image(image)
                if needs_cleanup:
                    temp_files.append(image_path)
                
                pil_image = Image.open(image_path).convert("RGB")
                model_inputs.append({
                    "prompt": ocr_prompt,
                    "multi_modal_data": {"image": pil_image}
                })
            
            # Configure sampling (logprobs=1: see _recognize_vllm / H1)
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=self.config.max_tokens,
                logprobs=1,
                extra_args={
                    "ngram_size": self.config.vllm_ngram_size,
                    "window_size": self.config.vllm_window_size,
                    "whitelist_token_ids": {128821, 128822},
                },
                skip_special_tokens=False,
            )
            
            # Batch inference
            outputs = self._vllm_model.generate(model_inputs, sampling_params)
            
            # Process results
            results = []
            for output in outputs:
                raw_text = output.outputs[0].text if output.outputs else ""
                model_conf = self._vllm_sequence_confidence(output)
                vin_text = self._extract_vin(raw_text)
                confidence = self._calculate_confidence(
                    vin_text, raw_text,
                    model_logits={"confidence": model_conf} if model_conf is not None else None,
                )
                
                results.append(OCRResult(
                    text=vin_text if vin_text else raw_text.upper().strip(),
                    confidence=confidence,
                    raw_response=raw_text,
                    provider=self.name,
                    metadata={
                        "backend": "vllm",
                        "batch": True,
                        "confidence_source": (
                            "model_logprobs" if model_conf is not None else "format_heuristic"
                        ),
                        "model_confidence": model_conf,
                    }
                ))
            
            return results
            
        finally:
            # Cleanup temp files
            for temp_path in temp_files:
                self._cleanup_temp(temp_path)
    
    def _vllm_sequence_confidence(self, request_output: Any) -> Optional[float]:
        """
        Compute the model's own sequence confidence from vLLM logprobs.

        Uses the geometric mean of the chosen tokens' probabilities
        (exp of the mean logprob), which is length-normalized and lives in
        (0, 1]. Returns None when logprobs are unavailable (older vLLM,
        logprobs not requested, or an API change), in which case the caller
        falls back to the format heuristic and says so in metadata.
        """
        try:
            completion = request_output.outputs[0]
            token_ids = completion.token_ids
            logprob_dicts = completion.logprobs
            if not logprob_dicts or not token_ids:
                return None

            chosen_logprobs: List[float] = []
            # strict: a token/logprob length mismatch means the vLLM API
            # changed shape - fall back to the heuristic (via the except
            # below) instead of silently computing over a truncated pair.
            for token_id, step in zip(token_ids, logprob_dicts, strict=True):
                if step is None or token_id not in step:
                    continue
                entry = step[token_id]
                # vLLM >= 0.4 wraps values in a Logprob object; older
                # versions used bare floats. Support both.
                chosen_logprobs.append(
                    float(entry.logprob if hasattr(entry, "logprob") else entry)
                )

            if not chosen_logprobs:
                return None
            mean_logprob = sum(chosen_logprobs) / len(chosen_logprobs)
            return float(min(1.0, max(0.0, np.exp(mean_logprob))))
        except (AttributeError, IndexError, KeyError, TypeError, ValueError) as e:
            logger.debug(f"Could not extract vLLM logprobs for confidence: {e}")
            return None

    def _extract_text(self, result: Any) -> str:
        """Extract text from model output."""
        if isinstance(result, str):
            return result
        elif isinstance(result, dict):
            return result.get('text', result.get('output', str(result)))
        elif isinstance(result, (list, tuple)) and result:
            return self._extract_text(result[0])
        return str(result) if result else ""
    
    def _extract_vin(self, text: str) -> str:
        """
        Extract the best VIN candidate from OCR text.

        Delegates to core.vin_utils.extract_vin_from_text, which scores EVERY
        17-character window (known-WMI bonus, decisive ISO 3779 checksum
        bonus) instead of taking the first regex match. The first-match
        approach shifted the VIN by one for any leading artifact character
        and silently dropped its last character (logic-audit M5).

        Returns:
            A 17-character VIN-charset string, or "" when no plausible
            candidate exists.
        """
        if not text:
            return ""

        # Clean obvious non-VIN noise; keep letters (a VIN may legitimately
        # start with X/Y/T/F/A, so nothing alphabetic is stripped here).
        cleaned = text.strip().replace('`', '').replace('*', '').replace('\n', ' ').upper()
        cleaned = ''.join(cleaned.split())
        if len(cleaned) < VINConstants.LENGTH:
            return ""

        candidate = extract_vin_from_text(cleaned)
        if len(candidate) != VINConstants.LENGTH:
            return ""

        if all(c in VINConstants.VALID_CHARS for c in candidate):
            return candidate

        # Window may carry I/O/Q misreads; apply the documented policy
        # (I->1, O->0, Q->0) and accept only if that makes it charset-valid.
        mapped = candidate.translate(str.maketrans("IOQ", "100"))
        if all(c in VINConstants.VALID_CHARS for c in mapped):
            return mapped
        return ""
    
    def _calculate_confidence(
        self,
        vin_text: str,
        raw_text: str,
        model_logits: Optional[Any] = None
    ) -> float:
        """
        Calculate confidence score using MODEL OUTPUT when available.
        
        Two regimes, deliberately distinct:
        1. Model confidence available (vLLM logprobs, CTC softmax, ...):
           format/checksum tiers are scaled by the model's own probability.
        2. No model signal: a FORMAT HEURISTIC. Being well-formed is weak
           evidence of being correct (wrong OCR reads are usually well-formed
           too), so the format-only value sits BELOW the typical softmax of a
           genuine engine and cannot outrank a calibrated competitor in an
           ensemble. A valid ISO 3779 check digit is strong evidence
           (a random wrong-but-well-formed read passes with p~=1/11), so it
           earns a high - but sub-calibrated-band - value.
        
        Confidence Tiers:
        - 0.95-1.00: Valid checksum + model confidence
        - 0.70-0.95: Valid format + model confidence
        - 0.90:      Valid checksum, no model signal (heuristic)
        - 0.50:      Valid format only, no model signal (heuristic)
        - 0.30-0.50: 17 chars but invalid characters
        - 0.10-0.30: Wrong length (symmetric distance from 17)
        - 0.00: Empty or failed
        
        Args:
            vin_text: Extracted VIN string
            raw_text: Raw OCR output
            model_logits: Optional model output (logits/probabilities/scores)
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not vin_text and not raw_text:
            return 0.0
        
        text = vin_text or raw_text.upper().strip()
        
        # Base confidence from model logits (if available)
        model_confidence = self._extract_model_confidence(model_logits)
        
        # Length check
        if len(text) != VINConstants.LENGTH:
            # Wrong length: 0.10-0.30, decreasing with SYMMETRIC distance from
            # 17. The previous min(len,17)/17 formula scored every too-long
            # text at the band maximum, outranking a 16-char near miss.
            length_ratio = max(
                0.0, 1.0 - abs(len(text) - VINConstants.LENGTH) / VINConstants.LENGTH
            )
            return 0.10 + (0.20 * length_ratio)
        
        # Character validity check
        invalid_chars = [c for c in text if c not in VINConstants.VALID_CHARS]
        if invalid_chars:
            # Has invalid chars: 0.30-0.50 based on how many
            valid_ratio = (VINConstants.LENGTH - len(invalid_chars)) / VINConstants.LENGTH
            return 0.30 + (0.20 * valid_ratio)
        
        # Valid format - now check checksum
        try:
            from src.vin_ocr.core.vin_utils import validate_checksum
            checksum_valid = validate_checksum(text)
        except ImportError:
            checksum_valid = False
        
        # Combine format validity with model confidence
        if model_confidence is not None:
            if checksum_valid:
                # Valid checksum + model confidence: 0.95-1.00
                return 0.95 + (0.05 * model_confidence)
            else:
                # Valid format + model confidence: 0.70-0.95
                return 0.70 + (0.25 * model_confidence)
        else:
            # No model confidence available - format heuristic (uncalibrated).
            # These values are intentionally below the model-confidence bands
            # so a fabricated constant can never beat a genuine engine score
            # in ensemble comparison (logic-audit H1).
            if checksum_valid:
                return 0.90  # Valid checksum: strong structural evidence
            else:
                return 0.50  # Valid format only: weak evidence
    
    def _extract_model_confidence(self, logits: Optional[Any]) -> Optional[float]:
        """
        Extract confidence from model logits/probabilities.
        
        For CTC-based models, this computes the average character probability.
        For VLM models, this extracts token probabilities.
        """
        if logits is None:
            return None
        
        try:
            import numpy as np
            
            # Handle different logit formats
            if hasattr(logits, 'numpy'):
                logits = logits.numpy()
            
            if isinstance(logits, np.ndarray):
                # CTC logits: [T, vocab_size] or [B, T, vocab_size]
                if logits.ndim == 3:
                    logits = logits[0]  # Take first batch
                
                # Softmax to get probabilities
                probs = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
                probs = probs / np.sum(probs, axis=-1, keepdims=True)
                
                # Get max probability per timestep
                max_probs = np.max(probs, axis=-1)
                
                # Average confidence (exclude very low values as padding)
                valid_mask = max_probs > 0.1
                if np.any(valid_mask):
                    return float(np.mean(max_probs[valid_mask]))
            
            # Handle dict with confidence key
            if isinstance(logits, dict):
                if 'confidence' in logits:
                    return float(logits['confidence'])
                if 'score' in logits:
                    return float(logits['score'])
            
            # Handle list of token probs
            if isinstance(logits, (list, tuple)):
                if all(isinstance(x, (int, float)) for x in logits):
                    return float(np.mean(logits))
            
        except Exception as e:
            logger.debug(f"Could not extract model confidence: {e}")
        
        return None
    
    def _cleanup_temp(self, path: str) -> None:
        """Safely cleanup temporary file with proper error logging."""
        try:
            import os
            if os.path.exists(path):
                os.unlink(path)
        except PermissionError as e:
            logger.warning(f"Permission denied cleaning up temp file {path}: {e}")
        except Exception as e:
            logger.debug(f"Could not cleanup temp file {path}: {e}")
    
    def _clean_response(self, text: str) -> str:
        """Legacy method - redirects to _extract_vin."""
        vin = self._extract_vin(text)
        return vin if vin else text.upper().strip()
    
    def _estimate_confidence(self, text: str) -> float:
        """Legacy method - redirects to _calculate_confidence."""
        return self._calculate_confidence(text, text)


# =============================================================================
# PROVIDER FACTORY
# =============================================================================

class OCRProviderFactory:
    """
    Factory for creating OCR provider instances.
    
    Usage:
        provider = OCRProviderFactory.create(OCRProviderType.PADDLEOCR)
        provider = OCRProviderFactory.create("deepseek", model_name="deepseek-ai/DeepSeek-OCR")
    """
    
    # Registry of available providers (properly typed)
    _providers: Dict[OCRProviderType, Type[OCRProvider]] = {
        OCRProviderType.PADDLEOCR: PaddleOCRProvider,
        OCRProviderType.DEEPSEEK: DeepSeekOCRProvider,
    }
    
    @classmethod
    def create(
        cls,
        provider_type: Union[str, OCRProviderType],
        auto_initialize: bool = True,
        **kwargs
    ) -> OCRProvider:
        """
        Create an OCR provider instance.
        
        Args:
            provider_type: Type of provider to create
            auto_initialize: Whether to initialize immediately
            **kwargs: Provider-specific configuration options
            
        Returns:
            Configured OCRProvider instance
            
        Raises:
            ValueError: If provider type is not supported
        """
        # Normalize provider type
        if isinstance(provider_type, str):
            try:
                provider_type = OCRProviderType(provider_type.lower())
            except ValueError:
                available = [p.value for p in OCRProviderType]
                raise ValueError(
                    f"Unknown provider type: '{provider_type}'. "
                    f"Available: {available}"
                )
        
        # Get provider class
        provider_class = cls._providers.get(provider_type)
        if provider_class is None:
            raise ValueError(f"Provider not implemented: {provider_type.value}")
        
        # Create config based on provider type
        config = cls._create_config(provider_type, **kwargs)
        
        # Create provider instance
        provider = provider_class(config=config)
        
        # Initialize if requested
        if auto_initialize and provider.is_available:
            provider.initialize()
        
        return provider
    
    # Accepted for backward compatibility and IGNORED with a warning:
    # DeepSeek-OCR runs locally and never needed an API key, but older docs
    # and call sites pass one.
    _LEGACY_IGNORED_KWARGS = frozenset({"api_key"})

    @classmethod
    def _create_config(
        cls,
        provider_type: OCRProviderType,
        **kwargs
    ) -> ProviderConfig:
        """
        Create provider-specific config from kwargs.

        Every kwarg must be a real field of the target config dataclass
        (including inherited ProviderConfig fields such as timeout,
        max_retries, retry_delay and the preprocess_* flags). Unknown kwargs
        raise ValueError instead of being silently dropped - silent dropping
        made typos invisible and made preprocess_enabled impossible to set
        through the factory (logic-audit L20). Defaults come from the
        dataclass itself, so there is a single source of truth (L19: the
        factory used to override max_tokens=128 against the documented 8192
        default, truncating output before the VIN appeared).
        """
        config = get_config()
        if provider_type == OCRProviderType.PADDLEOCR:
            config_cls: Type[ProviderConfig] = PaddleOCRConfig
            provider_defaults: Dict[str, Any] = {
                "lang": config.ocr.language,
                "use_gpu": config.ocr.use_gpu,
                "det_db_box_thresh": config.ocr.det_db_box_thresh,
                "rec_thresh": config.ocr.rec_thresh,
            }
        elif provider_type == OCRProviderType.DEEPSEEK:
            config_cls = DeepSeekOCRConfig
            provider_defaults = {
                "use_gpu": config.ocr.use_gpu,
            }
        else:
            config_cls = ProviderConfig
            provider_defaults = {}

        field_names = {f.name for f in dataclasses.fields(config_cls)}

        ignored = cls._LEGACY_IGNORED_KWARGS & set(kwargs)
        for name in sorted(ignored):
            logger.warning(
                f"Ignoring legacy kwarg '{name}' for {provider_type.value}: "
                "all registered providers run locally and require no API key"
            )

        unknown = set(kwargs) - field_names - cls._LEGACY_IGNORED_KWARGS
        if unknown:
            raise ValueError(
                f"Unknown option(s) for {provider_type.value} provider: "
                f"{sorted(unknown)}. Valid options: {sorted(field_names)}"
            )

        merged = dict(provider_defaults)
        merged.update({k: v for k, v in kwargs.items() if k in field_names})
        return config_cls(**merged)
    
    @classmethod
    def valid_options(cls, provider_type: Union[str, OCRProviderType]) -> frozenset:
        """
        Return the set of config option names a provider type accepts.

        Used by multi-provider callers (e.g. ensembles) to route per-provider
        options; single-provider creation validates strictly in _create_config.
        """
        if isinstance(provider_type, str):
            provider_type = OCRProviderType(provider_type.lower())
        config_cls: Type[ProviderConfig] = {
            OCRProviderType.PADDLEOCR: PaddleOCRConfig,
            OCRProviderType.DEEPSEEK: DeepSeekOCRConfig,
        }.get(provider_type, ProviderConfig)
        return frozenset(f.name for f in dataclasses.fields(config_cls))

    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered provider types."""
        return [p.value for p in cls._providers.keys()]
    
    @classmethod
    def register(
        cls,
        provider_type: OCRProviderType,
        provider_class: type
    ) -> None:
        """
        Register a new provider type.
        
        Args:
            provider_type: The provider type enum value
            provider_class: The provider class to register
        """
        if not issubclass(provider_class, OCRProvider):
            raise TypeError(
                f"Provider class must inherit from OCRProvider, "
                f"got {provider_class.__name__}"
            )
        cls._providers[provider_type] = provider_class
        logger.info(f"Registered OCR provider: {provider_type.value}")


# =============================================================================
# MULTI-PROVIDER ENSEMBLE
# =============================================================================

class EnsembleOCRProvider(OCRProvider):
    """
    Ensemble provider that combines results from multiple OCR backends.
    
    Strategies:
    - 'best': Return result with highest confidence
    - 'vote': Return most common result (majority voting)
    - 'weighted_vote': Return result with highest summed confidence
    - 'weighted_char_vote': Return per-character weighted vote
    - 'cascade': Try providers in order until valid result
    
    Usage:
        ensemble = EnsembleOCRProvider([
            OCRProviderFactory.create("paddleocr"),
            OCRProviderFactory.create("deepseek"),
        ], strategy='best')
        result = ensemble.recognize(image)
    """
    
    def __init__(
        self,
        providers: List[OCRProvider],
        strategy: str = "best"
    ):
        """
        Initialize ensemble provider.
        
        Args:
            providers: List of OCR providers to ensemble
            strategy: Ensemble strategy ('best', 'vote', 'cascade')
        """
        if not providers:
            raise ValueError("At least one provider is required")
        
        self.providers = providers
        self.strategy = strategy
        self._initialized = all(p.is_initialized for p in providers)
    
    @property
    def name(self) -> str:
        provider_names = [p.name for p in self.providers]
        return f"Ensemble({', '.join(provider_names)})"
    
    @property
    def is_available(self) -> bool:
        return any(p.is_available for p in self.providers)
    
    def initialize(self) -> None:
        """Initialize all providers."""
        for provider in self.providers:
            if provider.is_available and not provider.is_initialized:
                provider.initialize()
        self._initialized = True
    
    def recognize(
        self,
        image: Union[str, Path, np.ndarray],
        **kwargs
    ) -> OCRResult:
        """
        Recognize text using ensemble of providers.
        
        Args:
            image: Image path or numpy array
            **kwargs: Provider-specific options
            
        Returns:
            OCRResult from ensemble strategy
        """
        if not self._initialized:
            self.initialize()
        
        # Collect results from all providers
        results: List[OCRResult] = []
        errors: List[str] = []
        
        for provider in self.providers:
            if not provider.is_available:
                continue
            try:
                recognize_fn = getattr(provider, "recognize_with_retry", provider.recognize)
                result = recognize_fn(image, **kwargs)
                results.append(result)
            except Exception as e:
                errors.append(f"{provider.name}: {e}")
                logger.warning(f"Provider {provider.name} failed: {e}")
        
        if not results:
            raise OCRProviderError(
                f"All providers failed: {errors}",
                provider=self.name
            )
        
        # Apply ensemble strategy
        if self.strategy == "best":
            return self._best_strategy(results)
        elif self.strategy == "vote":
            return self._vote_strategy(results)
        elif self.strategy == "weighted_vote":
            return self._weighted_vote_strategy(results)
        elif self.strategy == "weighted_char_vote":
            return self._weighted_char_vote_strategy(results)
        elif self.strategy == "cascade":
            return self._cascade_strategy(results)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _best_strategy(self, results: List[OCRResult]) -> OCRResult:
        """Return result with highest confidence."""
        return max(results, key=lambda r: r.confidence)
    
    def _vote_strategy(self, results: List[OCRResult]) -> OCRResult:
        """
        Return most common result (majority voting).

        Ties are broken by total confidence, not by insertion order:
        Counter.most_common(1) alone returns the FIRST-inserted text on a
        tie, so two disagreeing providers always "elected" provider[0] and
        the vote never actually happened (logic-audit M3).
        """
        from collections import Counter
        
        texts = [r.text for r in results]
        counter = Counter(texts)
        top_count = max(counter.values())
        tied_texts = [t for t, c in counter.items() if c == top_count]
        
        tie_broken = len(tied_texts) > 1
        if tie_broken:
            most_common_text = max(
                tied_texts,
                key=lambda t: sum(
                    float(r.confidence or 0.0) for r in results if r.text == t
                ),
            )
        else:
            most_common_text = tied_texts[0]
        
        # Return the result with the most common text and highest confidence
        matching = [r for r in results if r.text == most_common_text]
        winner = max(matching, key=lambda r: r.confidence)
        
        # Add voting metadata
        winner.metadata["votes"] = dict(counter)
        winner.metadata["strategy"] = "vote"
        winner.metadata["tie_broken_by_confidence"] = tie_broken
        
        return winner

    def _weighted_vote_strategy(self, results: List[OCRResult]) -> OCRResult:
        """Return result with highest summed confidence across providers."""
        from collections import defaultdict

        weights: Dict[str, float] = defaultdict(float)
        for result in results:
            weights[result.text] += float(result.confidence or 0.0)

        best_text = max(weights.items(), key=lambda item: item[1])[0]
        matching = [r for r in results if r.text == best_text]
        winner = max(matching, key=lambda r: r.confidence)

        winner.metadata["weights"] = dict(weights)
        winner.metadata["strategy"] = "weighted_vote"
        return winner

    def _weighted_char_vote_strategy(self, results: List[OCRResult]) -> OCRResult:
        """
        Per-character weighted vote, using sequence ALIGNMENT.

        Candidates are aligned to the highest-confidence result before voting,
        rather than compared by raw index.

        Why: with naive positional voting, a candidate carrying a single
        leading artifact ("*SAL1..." vs "SAL1...") has every one of its
        characters shifted by one, so it votes the wrong character into every
        position. Given inputs

            A conf=0.80  SAL1A2A40SA606662   <- exactly correct
            B conf=0.75  *SAL1A2A40SA60666
            C conf=0.70  XSAL1A2A40SA60666

        positional voting returned "SSAL1A2A40SA60666" (edit distance 2) even
        though one input was perfect. An ensemble that can score worse than its
        best member is worse than useless, and leading artifacts are the
        documented dominant failure mode for these plates.

        Confidence is the weighted agreement FRACTION at each position
        (winning weight / total weight), averaged. The previous version summed
        raw confidences, so any three providers with confidence >= ~0.34
        saturated the clamp and reported 1.0 regardless of actual agreement.
        """
        from collections import defaultdict

        if not results:
            raise OCRProviderError("No results for weighted char vote", provider=self.name)

        usable = [r for r in results if r.text]
        if not usable:
            return self._best_strategy(results)

        best_result = self._best_strategy(results)

        # Anchor on the most confident candidate; prefer one of correct VIN
        # length if available, since that is the expected output shape.
        correct_len = [r for r in usable if len(r.text) == VINConstants.LENGTH]
        anchor = max(correct_len or usable, key=lambda r: r.confidence or 0.0)
        target_length = len(anchor.text)
        if target_length == 0:
            return self._best_strategy(results)

        char_weights: List[Dict[str, float]] = [defaultdict(float) for _ in range(target_length)]
        total_weights: List[float] = [0.0] * target_length

        for result in usable:
            weight = float(result.confidence or 0.0)
            if weight <= 0.0:
                continue

            if result is anchor:
                votes = {i: ch for i, ch in enumerate(result.text)}
            else:
                # Map anchor position -> this candidate's aligned character.
                votes = {}
                matcher = SequenceMatcher(None, anchor.text, result.text, autojunk=False)
                for tag, a0, a1, r0, r1 in matcher.get_opcodes():
                    if tag == 'equal':
                        for offset in range(a1 - a0):
                            votes[a0 + offset] = result.text[r0 + offset]
                    elif tag == 'replace':
                        # Align the overlapping prefix of the replaced spans
                        for offset in range(min(a1 - a0, r1 - r0)):
                            votes[a0 + offset] = result.text[r0 + offset]
                    # 'delete' -> candidate has no character for these anchor
                    # positions; 'insert' -> extra characters with no anchor
                    # position. Both correctly contribute no vote.

            for idx, ch in votes.items():
                if 0 <= idx < target_length:
                    char_weights[idx][ch] += weight
                    total_weights[idx] += weight

        chosen_chars: List[str] = []
        agreement_fractions: List[float] = []
        for idx, weights in enumerate(char_weights):
            if not weights:
                chosen_chars.append(anchor.text[idx] if idx < len(anchor.text) else "")
                agreement_fractions.append(0.0)
                continue

            best_char, best_weight = max(weights.items(), key=lambda item: item[1])
            chosen_chars.append(best_char)
            total = total_weights[idx]
            agreement_fractions.append(best_weight / total if total > 0 else 0.0)

        merged_text = ''.join(chosen_chars)

        # Two independent signals, deliberately kept separable:
        #   agreement       - how strongly the providers concur, 0..1
        #   provider_conf   - how confident the providers were to begin with
        #
        # Reporting agreement alone saturates: three providers that are each
        # only 34% sure but happen to concur would report 1.0. Reporting raw
        # confidence alone ignores consensus. The product is a documented
        # HEURISTIC, not a calibrated probability - calibrating it properly
        # needs labelled data, which is not available here. Both components are
        # exposed in metadata so downstream code can apply its own policy.
        agreement = (
            sum(agreement_fractions) / target_length if target_length else 0.0
        )
        contributing = [float(r.confidence or 0.0) for r in usable if (r.confidence or 0.0) > 0.0]
        provider_conf = sum(contributing) / len(contributing) if contributing else 0.0

        confidence = max(0.0, min(1.0, agreement * provider_conf))

        return OCRResult(
            text=merged_text,
            confidence=confidence,
            provider=self.name,
            metadata={
                "strategy": "weighted_char_vote",
                "confidence_model": "agreement * mean_provider_confidence (heuristic, uncalibrated)",
                "agreement": agreement,
                "mean_provider_confidence": provider_conf,
                "char_weights": [dict(w) for w in char_weights],
                "anchor": anchor.text,
                "fallback": best_result.text,
            },
        )
    
    def _cascade_strategy(self, results: List[OCRResult]) -> OCRResult:
        """Return first valid result (cascade through providers)."""
        for result in results:
            # Check if result looks valid (VIN length, valid VIN characters)
            if len(result.text) == VINConstants.LENGTH:
                if all(c in VINConstants.VALID_CHARS for c in result.text):
                    result.metadata["strategy"] = "cascade"
                    return result
        
        # No valid result, return best confidence
        return self._best_strategy(results)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_default_provider() -> OCRProvider:
    """Get the default OCR provider (PaddleOCR)."""
    return OCRProviderFactory.create(OCRProviderType.PADDLEOCR)


def recognize_vin(
    image: Union[str, Path, np.ndarray],
    provider: Optional[Union[str, OCRProvider]] = None,
    **kwargs
) -> OCRResult:
    """
    Convenience function to recognize VIN from image.
    
    Args:
        image: Image path or numpy array
        provider: Provider type string or OCRProvider instance
        **kwargs: Provider configuration options
        
    Returns:
        OCRResult with recognized VIN
        
    Example:
        result = recognize_vin("image.jpg")
        result = recognize_vin("image.jpg", provider="deepseek")
    """
    if provider is None:
        ocr = get_default_provider()
    elif isinstance(provider, str):
        ocr = OCRProviderFactory.create(provider, **kwargs)
    else:
        ocr = provider
    
    return ocr.recognize(image)
