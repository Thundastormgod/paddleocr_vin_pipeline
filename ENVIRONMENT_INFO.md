🌍 **Conda Environment Location**
=====================================

**Environment Name:** `paddleocr_vin`

**Full Path:** 
```
C:\Users\OVST\miniconda3\envs\paddleocr_vin
```

**Python Executable:**
```
C:\Users\OVST\miniconda3\envs\paddleocr_vin\python.exe
```

**Activate Environment:**
```bash
conda activate paddleocr_vin
```

**Quick Start Script:**
```
E:\paddle and deepseek OCR\paddleocr_vin_pipeline\run_streamlit.bat
```

📦 **Installed Key Dependencies**
=====================================

✅ **PaddlePaddle:** 3.0.0 (CPU)
✅ **PyTorch:** 2.7.1+cu118 (CUDA 11.8)
✅ **Transformers:** 5.0.0
✅ **PaddleOCR:** 3.0.0
✅ **PEFT:** 0.18.1 (LoRA fine-tuning)
✅ **bitsandbytes:** 0.49.1 (Quantization)
✅ **Streamlit:** Latest

**DeepSeek-OCR Dependencies:**
✅ addict, matplotlib, einops, easydict, timm

🔧 **Install/Update Commands**
=====================================

**Install all dependencies:**
```bash
conda activate paddleocr_vin
pip install -r requirements_complete.txt
```

**Install PyTorch with CUDA:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Install PaddlePaddle GPU:**
```bash
pip uninstall paddlepaddle -y
pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu118/
```

**Freeze current dependencies:**
```bash
pip freeze > requirements_frozen.txt
```

📝 **Files Created**
=====================================

1. `requirements.txt` - Original requirements (kept for compatibility)
2. `requirements_complete.txt` - **Complete single source of truth**
3. `requirements_frozen.txt` - Exact versions from pip freeze
4. `run_streamlit.bat` - Quick launcher with environment activation
5. `ENVIRONMENT_INFO.md` - This file

🚀 **Usage**
=====================================

**Option 1: Use batch script (Recommended)**
```
run_streamlit.bat
```

**Option 2: Manual activation**
```bash
conda activate paddleocr_vin
cd "E:\paddle and deepseek OCR\paddleocr_vin_pipeline"
python -m streamlit run web_ui/app.py
```

**Option 3: Direct execution**
```bash
C:\Users\OVST\miniconda3\envs\paddleocr_vin\python.exe -m streamlit run web_ui/app.py
```
