# =============================================================================
# VIN OCR Pipeline - Universal Dockerfile (Auto-Detect Hardware)
# =============================================================================
# This is the main Dockerfile that works on any platform.
# It uses CPU by default but can leverage GPU when available at runtime.
#
# For optimal GPU performance, use:
#   - docker/Dockerfile.gpu for NVIDIA CUDA
#   - docker/Dockerfile.cpu for CPU-only deployments
#
# Build: docker build -t vin-ocr .
# Run:   docker run -p 8501:8501 vin-ocr
# =============================================================================

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

LABEL maintainer="VIN OCR Team"
LABEL version="1.0.0"
LABEL description="VIN OCR Pipeline - Universal (Auto-detect hardware)"

WORKDIR /app

# Install runtime dependencies for OpenCV and image processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY config.py ./config.py
COPY docker/entrypoint.sh ./entrypoint.sh

# Make entrypoint executable
RUN chmod +x ./entrypoint.sh

# Create necessary directories
RUN mkdir -p output data dataset models logs finetune_data

# Create non-root user for security
RUN useradd -m -u 1000 vinuser && \
    chown -R vinuser:vinuser /app
USER vinuser

# Expose Streamlit port
EXPOSE 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Entrypoint for automatic hardware detection
ENTRYPOINT ["./entrypoint.sh"]

# Default command - run Streamlit
CMD ["streamlit", "run", "src/vin_ocr/web/app.py"]
