# Contributing to VIN OCR Pipeline

Thank you for your interest in contributing to the VIN OCR Pipeline! This document provides guidelines and instructions for contributing.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Making Changes](#making-changes)
- [Pull Request Process](#pull-request-process)
- [Coding Standards](#coding-standards)
- [Testing](#testing)

## Code of Conduct

Please be respectful and constructive in all interactions. We welcome contributions from everyone regardless of experience level.

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/paddleocr_vin_pipeline.git
   cd paddleocr_vin_pipeline
   ```
3. Add the upstream remote:
   ```bash
   git remote add upstream https://github.com/Thundastormgod/paddleocr_vin_pipeline.git
   ```

## Development Setup

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install development dependencies:
   ```bash
   make dev
   # Or manually:
   pip install -e ".[all]"
   pre-commit install
   ```

3. Verify installation:
   ```bash
   make validate
   make test
   ```

## Making Changes

1. Create a new branch:
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

2. Make your changes following the [coding standards](#coding-standards)

3. Write tests for new functionality

4. Run the test suite:
   ```bash
   make test
   ```

5. Run linting:
   ```bash
   make lint
   ```

6. Format code:
   ```bash
   make format
   ```

## Pull Request Process

1. Update documentation if needed
2. Add your changes to CHANGELOG.md under "Unreleased"
3. Ensure all tests pass
4. Push to your fork and create a Pull Request
5. Fill out the PR template completely
6. Wait for review and address any feedback

## Coding Standards

### Python Style

- Follow PEP 8
- Use Black for formatting (line length: 100)
- Use isort for import sorting
- Add type hints where possible
- Write docstrings for all public functions/classes

### Example:

```python
def recognize_vin(
    image_path: str,
    model: Optional[str] = None,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Recognize VIN from an image.
    
    Args:
        image_path: Path to the input image file.
        model: Optional path to custom model.
        confidence_threshold: Minimum confidence for valid prediction.
        
    Returns:
        Dictionary containing:
        - vin: Recognized VIN string
        - confidence: Prediction confidence (0-1)
        - raw_text: Raw OCR output
        
    Raises:
        FileNotFoundError: If image_path doesn't exist.
        ValueError: If image cannot be processed.
    """
    ...
```

### Commit Messages

Use conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `style:` Code style (formatting, etc.)
- `refactor:` Code refactoring
- `test:` Adding tests
- `chore:` Maintenance tasks

Example: `feat: add batch inference support for ONNX models`

## Testing

### Running Tests

```bash
# All tests
make test

# With coverage
make test-cov

# Specific test file
pytest tests/test_inference.py -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Name test functions `test_*`
- Use pytest fixtures for setup
- Mock external dependencies

Example:

```python
import pytest
from src.vin_ocr.inference import VINInference

class TestVINInference:
    def test_recognize_returns_valid_result(self, sample_image):
        inference = VINInference("output/model/inference")
        result = inference.recognize(sample_image)
        
        assert 'vin' in result
        assert 'confidence' in result
        assert result['error'] is None
```

## Questions?

Feel free to open an issue for questions or discussions!
