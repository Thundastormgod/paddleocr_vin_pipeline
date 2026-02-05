.PHONY: help install dev test lint format clean run train evaluate docker

# Default target
help:
	@echo "VIN OCR Pipeline - Available commands:"
	@echo ""
	@echo "  make install     Install package in production mode"
	@echo "  make dev         Install package in development mode"
	@echo "  make test        Run tests"
	@echo "  make lint        Run linting checks"
	@echo "  make format      Format code with black and isort"
	@echo "  make clean       Clean build artifacts"
	@echo "  make run         Start the Streamlit web UI"
	@echo "  make train       Start training (interactive)"
	@echo "  make evaluate    Run model evaluation"
	@echo "  make docker      Build Docker image"
	@echo ""

# Installation
install:
	pip install -e .

dev:
	pip install -e ".[all]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --cov=src --cov-report=html --cov-report=term

# Linting and formatting
lint:
	flake8 src/ tests/
	mypy src/
	black --check src/ tests/
	isort --check-only src/ tests/

format:
	black src/ tests/
	isort src/ tests/

# Cleaning
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Running
run:
	streamlit run src/vin_ocr/web/app_simple.py --server.port 8501

run-full:
	streamlit run src/vin_ocr/web/app.py --server.port 8501

run-debug:
	streamlit run src/vin_ocr/web/app_simple.py --server.port 8501 --logger.level=debug

# Training
train:
	python -m src.vin_ocr.training.finetune_paddleocr --help

train-scratch:
	python -m src.vin_ocr.training.train_from_scratch --help

# Evaluation
evaluate:
	python -m src.vin_ocr.evaluation.evaluate --help

# ONNX conversion
convert-onnx:
	python scripts/reexport_and_convert_onnx.py

# Docker
docker:
	docker build -t vin-ocr-pipeline:latest .

docker-run:
	docker run -p 8501:8501 vin-ocr-pipeline:latest

# Data preparation
prepare-data:
	python scripts/prepare_dataset.py
	python scripts/prepare_finetune_data.py

# Quick validation
validate:
	python -c "from src.vin_ocr.inference import VINInference, ONNXVINRecognizer; print('✅ Imports OK')"
	python -c "import paddle; print(f'✅ PaddlePaddle {paddle.__version__}')"
	python -c "import onnxruntime; print(f'✅ ONNX Runtime {onnxruntime.__version__}')"
