# DVC + DagsHub Data Streaming Setup

> [!WARNING]
> **PROVENANCE CORRECTION (2026-08-18).** The streaming integration this
> document describes **does not exist in this repository**: it depends on
> `src/vin_ocr/data/` (never committed), and both entry points
> (`setup_dagshub_dvc.py`, `train_vin_streaming.py`) crash on import with
> `ModuleNotFoundError` - reproduced during audit. The `--stream` flag is
> never registered because its import guard always fails. Claims below of
> a "complete", "functional" or "production-ready" streaming setup are
> false. Only the plain DVC remote configuration (`.dvc/config`) is real.


This guide shows how to set up DVC and DagsHub data streaming for the VIN OCR project, allowing training without local data downloads.

## 🚀 Quick Start

### 1. First Time Setup

```bash
# Install dependencies
pip install dvc dagshub

# Run complete setup
python setup_dagshub_dvc.py \
  --repo-owner YOUR_USERNAME \
  --repo-name YOUR_REPO \
  --username YOUR_USERNAME \
  --token YOUR_DAGSHUB_TOKEN
```

### 2. Track Your Data

```bash
# Add data files to DVC tracking
dvc add finetune_data/train_images
dvc add finetune_data/val_images
dvc add finetune_data/train_labels.txt
dvc add finetune_data/val_labels.txt

# Commit DVC files
git add *.dvc .gitignore
git commit -m "Track datasets with DVC"
```

### 3. Push Everything

```bash
# Push data to DagsHub
dvc push

# Push code to DagsHub
git push origin main
```

### 4. Train with Streaming (No Local Data Needed)

```bash
# On training server or new machine
git clone https://dagshub.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Train directly with streaming
python train_vin_streaming.py --stream \
  --repo-owner YOUR_USERNAME \
  --repo-name YOUR_REPO \
  --dagshub-user YOUR_USERNAME \
  --dagshub-token YOUR_DAGSHUB_TOKEN \
  --epochs 10 --batch-size 16
```

## 📋 Detailed Setup

### Prerequisites

1. **DagsHub Account**: Create account at [dagshub.com](https://dagshub.com)
2. **Repository**: Create a new repository for your VIN OCR project
3. **Access Token**: Generate token in DagsHub settings

### Installation

```bash
# Install required packages
pip install dvc dagshub paddlepaddle

# Or install all dependencies
pip install -r requirements.txt
```

### Configuration

The setup script will:

1. ✅ Initialize DVC in your project
2. ✅ Configure DagsHub as DVC remote
3. ✅ Set up data streaming
4. ✅ Update configuration files
5. ✅ Create streaming-enabled training script

### Data Structure

Your project should have this structure:

```
paddleocr_vin_pipeline/
├── finetune_data/
│   ├── train_images/          # Training images
│   ├── val_images/            # Validation images
│   ├── train_labels.txt       # Training labels
│   └── val_labels.txt         # Validation labels
├── dagshub_data/              # Streaming data cache
├── .dvc/                      # DVC configuration
├── .dvcignore                 # DVC ignore file
└── setup_dagshub_dvc.py       # Setup script
```

## 🔄 Workflow Commands

### Local Development

```bash
# Add new data
dvc add data/new_images

# Push changes
dvc push
git add . && git commit -m "Add new data" && git push
```

### Remote Training

```bash
# Clone repository (no data downloaded)
git clone https://dagshub.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO

# Option 1: Pull all data locally
dvc pull
python train_vin_model.py

# Option 2: Stream directly (recommended)
python train_vin_streaming.py --stream \
  --repo-owner YOUR_USERNAME \
  --repo-name YOUR_REPO \
  --dagshub-user YOUR_USERNAME \
  --dagshub-token YOUR_TOKEN
```

### Environment Variables

You can also use environment variables instead of command line arguments:

```bash
export DAGSHUB_USERNAME="your_username"
export DAGSHUB_TOKEN="your_token"

python train_vin_streaming.py --stream \
  --repo-owner YOUR_USERNAME \
  --repo-name YOUR_REPO
```

## 📊 Streaming Benefits

### ✅ Advantages

- **No Local Storage**: Train without downloading GBs of images
- **Always Updated**: Always use latest data version
- **Collaborative**: Team members use same data
- **Version Control**: Track data changes with Git
- **Reproducible**: Exact same data for every experiment

### 🚀 Performance

- **Streaming**: Data loads on-demand during training
- **Caching**: Frequently accessed data cached locally
- **Parallel**: Multiple training jobs can stream simultaneously

## 🔧 Advanced Usage

### Custom Data Paths

```python
from src.vin_ocr.data.setup_dagshub_streaming import DagsHubDataStreamer

# Initialize custom streaming
streamer = DagsHubDataStreamer("username", "repo")
streamer.initialize_streaming("username", "token")

# Access specific files
image_path = streamer.get_data_path("finetune_data/train_images/image_001.jpg")
```

### Multiple Datasets

```bash
# Track different dataset versions
dvc add data/v1/train_images
dvc add data/v2/train_images

# Switch between versions
dvc checkout data/v1/train_images
```

### Partial Downloads

```bash
# Download only specific files
dvc pull finetune_data/train_labels.txt

# Download by directory
dvc pull finetune_data/val_images
```

## 🛠️ Troubleshooting

### Common Issues

1. **Authentication Error**
   ```bash
   # Check token
   dvc remote list
   dvc remote modify origin --local auth basic
   dvc remote modify origin --local user YOUR_USERNAME
   dvc remote modify origin --local password YOUR_TOKEN
   ```

2. **Streaming Not Working**
   ```bash
   # Check dagshub installation
   pip show dagshub
   
   # Reinstall if needed
   pip install --upgrade dagshub
   ```

3. **Data Not Found**
   ```bash
   # Check if data exists in remote
   dvc ls remote://origin/finetune_data
   ```

### Debug Mode

```bash
# Enable debug logging
export DVC_DEBUG=true
export DAGSHUB_DEBUG=true

python train_vin_streaming.py --stream ...
```

## 📚 Resources

- [DVC Documentation](https://dvc.org/doc)
- [DagsHub Documentation](https://dagshub.com/docs)
- [DVC + DagsHub Integration](https://dagshub.com/docs/dvc)

## 🤝 Contributing

To contribute to the streaming setup:

1. Fork the repository
2. Create feature branch
3. Test streaming functionality
4. Submit pull request

## 📄 License

This setup follows the same license as the VIN OCR project.
