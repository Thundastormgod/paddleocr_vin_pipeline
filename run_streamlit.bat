@echo off
REM Streamlit VIN OCR Application Launcher
REM Ensures all dependencies are loaded from paddleocr_vin conda environment

echo ========================================
echo Starting VIN OCR Streamlit Application
echo ========================================

REM Activate conda environment
call C:\Users\OVST\miniconda3\Scripts\activate.bat paddleocr_vin

REM Set environment variables
set DISABLE_MODEL_SOURCE_CHECK=True

REM Navigate to project directory
cd /d "E:\paddle and deepseek OCR\paddleocr_vin_pipeline"

REM Run Streamlit
echo Starting Streamlit on port 8501...
python -m streamlit run web_ui/app.py --server.port 8501

pause
