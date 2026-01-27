#!/usr/bin/env python3
"""
VIN OCR Web UI - Streamlit Application
======================================

A user-friendly web interface for VIN recognition using multiple OCR models.

Features:
- Model Selection: Choose from PaddleOCR v3/v4, VIN Pipeline, or DeepSeek-OCR
- Single Image Recognition: Upload and process individual images
- Batch Processing: Process entire folders of images
- Training Interface: Train/fine-tune models on custom datasets
- Results Dashboard: View metrics, comparisons, and export results

Usage:
    streamlit run web_ui/app.py
    
    # Or with custom port:
    streamlit run web_ui/app.py --server.port 8501

Requirements:
    pip install streamlit plotly pandas pillow

Author: JRL-VIN Project
Date: January 2026
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any

import streamlit as st
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Optional imports with graceful fallback
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Project imports
try:
    from vin_pipeline import VINOCRPipeline
    VIN_PIPELINE_AVAILABLE = True
except ImportError:
    VIN_PIPELINE_AVAILABLE = False

try:
    from multi_model_evaluation import MultiModelEvaluator, VINCharValidator
    EVALUATOR_AVAILABLE = True
except ImportError:
    EVALUATOR_AVAILABLE = False


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="VIN OCR Recognition System",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 1rem;
        margin: 1rem 0;
    }
    .vin-display {
        font-family: 'Courier New', monospace;
        font-size: 1.5rem;
        font-weight: bold;
        letter-spacing: 0.2rem;
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 5px;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = {}
    if 'results_history' not in st.session_state:
        st.session_state.results_history = []
    if 'current_model' not in st.session_state:
        st.session_state.current_model = None
    if 'training_status' not in st.session_state:
        st.session_state.training_status = None


# =============================================================================
# MODEL MANAGEMENT
# =============================================================================

@st.cache_resource
def load_vin_pipeline():
    """Load VIN Pipeline (cached)."""
    if VIN_PIPELINE_AVAILABLE:
        return VINOCRPipeline()
    return None


@st.cache_resource
def load_paddleocr(version: str = "v4"):
    """Load PaddleOCR model (cached)."""
    try:
        from paddleocr import PaddleOCR
        if version == "v3":
            return PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                ocr_version='PP-OCRv3',
            )
        else:
            return PaddleOCR(
                use_textline_orientation=True,
                lang='en',
                text_det_thresh=0.3,
                text_det_box_thresh=0.5,
            )
    except Exception as e:
        st.error(f"Failed to load PaddleOCR {version}: {e}")
        return None


@st.cache_resource
def load_deepseek():
    """Load DeepSeek-OCR model (cached)."""
    try:
        from ocr_providers import DeepSeekOCRProvider
        provider = DeepSeekOCRProvider()
        if provider.is_available:
            return provider
        else:
            st.warning("DeepSeek-OCR dependencies not installed")
            return None
    except Exception as e:
        st.error(f"Failed to load DeepSeek-OCR: {e}")
        return None


def get_available_models() -> Dict[str, bool]:
    """Check which models are available."""
    models = {
        "VIN Pipeline (PP-OCRv5)": VIN_PIPELINE_AVAILABLE,
        "PaddleOCR PP-OCRv4": True,  # Will check on load
        "PaddleOCR PP-OCRv3": True,  # Will check on load
        "DeepSeek-OCR": False,  # Will check on load
    }
    
    # Check DeepSeek availability
    try:
        from ocr_providers import DeepSeekOCRProvider
        provider = DeepSeekOCRProvider()
        models["DeepSeek-OCR"] = provider.is_available
    except:
        pass
    
    return models


# =============================================================================
# RECOGNITION FUNCTIONS
# =============================================================================

def recognize_with_model(image_path: str, model_name: str) -> Dict[str, Any]:
    """Run recognition with selected model."""
    start_time = time.time()
    result = {
        'vin': '',
        'confidence': 0.0,
        'model': model_name,
        'processing_time': 0.0,
        'error': None
    }
    
    try:
        if model_name == "VIN Pipeline (PP-OCRv5)":
            pipeline = load_vin_pipeline()
            if pipeline:
                res = pipeline.recognize(image_path)
                result['vin'] = res.get('vin', '')
                result['confidence'] = res.get('confidence', 0.0)
                result['raw_ocr'] = res.get('raw_ocr', '')
                result['corrections'] = res.get('corrections', [])
        
        elif model_name.startswith("PaddleOCR"):
            version = "v3" if "v3" in model_name else "v4"
            engine = load_paddleocr(version)
            if engine:
                ocr_result = engine.predict(image_path)
                texts, confidences = [], []
                for item in ocr_result:
                    if hasattr(item, 'rec_texts'):
                        texts.extend(item.rec_texts or [])
                        confidences.extend(item.rec_scores or [])
                
                combined_text = ' '.join(texts)
                result['raw_ocr'] = combined_text
                result['confidence'] = sum(confidences) / len(confidences) if confidences else 0.0
                
                # Extract VIN
                if EVALUATOR_AVAILABLE:
                    result['vin'] = VINCharValidator.extract_vin_from_text(combined_text)
                else:
                    result['vin'] = ''.join(c for c in combined_text.upper() if c.isalnum())[:17]
        
        elif model_name == "DeepSeek-OCR":
            provider = load_deepseek()
            if provider:
                if not provider._initialized:
                    provider.initialize()
                res = provider.recognize(image_path)
                result['vin'] = res.text if res else ''
                result['confidence'] = res.confidence if res else 0.0
    
    except Exception as e:
        result['error'] = str(e)
    
    result['processing_time'] = time.time() - start_time
    return result


# =============================================================================
# UI COMPONENTS
# =============================================================================

def render_sidebar():
    """Render sidebar with navigation and settings."""
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/car-roof-box.png", width=80)
        st.title("VIN OCR System")
        
        st.divider()
        
        # Navigation
        page = st.radio(
            "Navigation",
            ["🔍 Recognition", "📊 Batch Evaluation", "🎯 Training", "📈 Results Dashboard"],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        # Model Selection
        st.subheader("🤖 Model Selection")
        available_models = get_available_models()
        
        model_options = [m for m, available in available_models.items()]
        selected_model = st.selectbox(
            "Choose Model",
            model_options,
            help="Select the OCR model to use for recognition"
        )
        
        # Show model status
        if available_models.get(selected_model, False):
            st.success(f"✓ {selected_model} available")
        else:
            st.warning(f"⚠ {selected_model} may need setup")
        
        st.session_state.current_model = selected_model
        
        st.divider()
        
        # Quick Stats
        st.subheader("📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Images Processed", len(st.session_state.results_history))
        with col2:
            if st.session_state.results_history:
                avg_conf = sum(r.get('confidence', 0) for r in st.session_state.results_history) / len(st.session_state.results_history)
                st.metric("Avg Confidence", f"{avg_conf:.1%}")
            else:
                st.metric("Avg Confidence", "N/A")
        
        return page


def render_recognition_page():
    """Render the single image recognition page."""
    st.markdown('<h1 class="main-header">🔍 VIN Recognition</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Upload an image to extract the Vehicle Identification Number</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("📤 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose a VIN image",
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image containing a VIN plate or sticker"
        )
        
        if uploaded_file:
            if PIL_AVAILABLE:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", width="stretch")
            else:
                st.info("Image preview not available (PIL not installed)")
    
    with col2:
        st.subheader("📋 Recognition Result")
        
        if uploaded_file:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            
            model_name = st.session_state.current_model
            
            with st.spinner(f"Processing with {model_name}..."):
                result = recognize_with_model(tmp_path, model_name)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Display results
            if result['error']:
                st.error(f"Error: {result['error']}")
            else:
                # VIN Display
                vin = result['vin'] or "No VIN detected"
                st.markdown(f'<div class="vin-display">{vin}</div>', unsafe_allow_html=True)
                
                # Metrics
                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.metric("Confidence", f"{result['confidence']:.1%}")
                with col_b:
                    st.metric("Length", f"{len(result['vin'])}/17")
                with col_c:
                    st.metric("Time", f"{result['processing_time']:.2f}s")
                
                # Validation
                if len(result['vin']) == 17:
                    st.success("✓ Valid VIN length (17 characters)")
                else:
                    st.warning(f"⚠ Invalid VIN length ({len(result['vin'])} characters)")
                
                # Raw OCR output
                with st.expander("🔍 Raw OCR Output"):
                    st.code(result.get('raw_ocr', 'N/A'))
                
                # Corrections applied
                if result.get('corrections'):
                    with st.expander("🔧 Corrections Applied"):
                        for corr in result['corrections']:
                            st.write(f"- {corr}")
                
                # Save to history
                result['timestamp'] = datetime.now().isoformat()
                result['filename'] = uploaded_file.name
                st.session_state.results_history.append(result)
        else:
            st.info("👆 Upload an image to get started")


def render_batch_evaluation_page():
    """Render the batch evaluation page."""
    st.markdown('<h1 class="main-header">📊 Batch Evaluation</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Evaluate models on multiple images with ground truth labels</p>', unsafe_allow_html=True)
    
    # Input options
    tab1, tab2 = st.tabs(["📁 Folder Path", "📤 Upload Files"])
    
    with tab1:
        st.subheader("Specify Image Folder")
        
        col1, col2 = st.columns(2)
        with col1:
            image_folder = st.text_input(
                "Image Folder Path",
                value="./dagshub_data/test/images",
                help="Path to folder containing VIN images"
            )
        with col2:
            labels_file = st.text_input(
                "Labels File (optional)",
                value="",
                help="Path to labels file (format: image_path\\tVIN)"
            )
        
        max_images = st.slider("Max Images to Process", 10, 500, 50)
        
        # Model selection for comparison
        st.subheader("Models to Evaluate")
        available_models = get_available_models()
        selected_models = st.multiselect(
            "Select models for comparison",
            list(available_models.keys()),
            default=["VIN Pipeline (PP-OCRv5)"]
        )
        
        if st.button("🚀 Run Evaluation", type="primary"):
            if not Path(image_folder).exists():
                st.error(f"Folder not found: {image_folder}")
            else:
                run_batch_evaluation(image_folder, labels_file, max_images, selected_models)
    
    with tab2:
        st.subheader("Upload Images")
        uploaded_files = st.file_uploader(
            "Choose VIN images",
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True
        )
        
        if uploaded_files:
            st.info(f"📁 {len(uploaded_files)} images uploaded")
            
            if st.button("🚀 Process Uploaded Images", type="primary"):
                process_uploaded_batch(uploaded_files)


def run_batch_evaluation(image_folder: str, labels_file: str, max_images: int, models: List[str]):
    """Run batch evaluation on a folder of images."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Find images
    folder = Path(image_folder)
    image_files = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
    image_files = image_files[:max_images]
    
    if not image_files:
        st.error("No images found in folder")
        return
    
    st.info(f"Found {len(image_files)} images")
    
    results_by_model = {model: [] for model in models}
    
    for i, img_path in enumerate(image_files):
        progress_bar.progress((i + 1) / len(image_files))
        status_text.text(f"Processing {img_path.name}...")
        
        for model in models:
            result = recognize_with_model(str(img_path), model)
            result['filename'] = img_path.name
            
            # Try to extract ground truth from filename
            gt_vin = extract_vin_from_filename(img_path.name)
            result['ground_truth'] = gt_vin
            result['exact_match'] = result['vin'] == gt_vin if gt_vin else None
            
            results_by_model[model].append(result)
    
    status_text.text("Evaluation complete!")
    
    # Display results
    display_evaluation_results(results_by_model)


def extract_vin_from_filename(filename: str) -> Optional[str]:
    """Extract VIN from filename pattern."""
    # Pattern: VIN_...-VIN_-_VIN_.jpg
    parts = filename.upper().replace('-', '_').split('_')
    for part in parts:
        cleaned = ''.join(c for c in part if c.isalnum())
        if len(cleaned) == 17 and not any(c in cleaned for c in 'IOQ'):
            return cleaned
    return None


def display_evaluation_results(results_by_model: Dict[str, List[Dict]]):
    """Display evaluation results with charts."""
    st.subheader("📈 Evaluation Results")
    
    # Summary metrics
    summary_data = []
    for model, results in results_by_model.items():
        valid_results = [r for r in results if r.get('ground_truth')]
        if valid_results:
            exact_matches = sum(1 for r in valid_results if r.get('exact_match'))
            char_correct = sum(
                sum(1 for a, b in zip(r['vin'][:17], r['ground_truth'][:17]) if a == b)
                for r in valid_results
            )
            total_chars = len(valid_results) * 17
            
            summary_data.append({
                'Model': model,
                'Total Images': len(results),
                'With Ground Truth': len(valid_results),
                'Exact Matches': exact_matches,
                'Exact Match %': exact_matches / len(valid_results) * 100 if valid_results else 0,
                'Character Accuracy %': char_correct / total_chars * 100 if total_chars else 0,
                'Avg Confidence': sum(r['confidence'] for r in results) / len(results) * 100,
                'Avg Time (s)': sum(r['processing_time'] for r in results) / len(results)
            })
        else:
            summary_data.append({
                'Model': model,
                'Total Images': len(results),
                'With Ground Truth': 0,
                'Exact Matches': 'N/A',
                'Exact Match %': 'N/A',
                'Character Accuracy %': 'N/A',
                'Avg Confidence': sum(r['confidence'] for r in results) / len(results) * 100 if results else 0,
                'Avg Time (s)': sum(r['processing_time'] for r in results) / len(results) if results else 0
            })
    
    # Display summary table
    df_summary = pd.DataFrame(summary_data)
    st.dataframe(df_summary, width="stretch")
    
    # Charts
    if PLOTLY_AVAILABLE and len(summary_data) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                df_summary,
                x='Model',
                y='Character Accuracy %',
                title='Character Accuracy by Model',
                color='Model'
            )
            st.plotly_chart(fig, width="stretch")
        
        with col2:
            fig = px.bar(
                df_summary,
                x='Model',
                y='Avg Time (s)',
                title='Processing Time by Model',
                color='Model'
            )
            st.plotly_chart(fig, width="stretch")
    
    # Detailed results expander
    with st.expander("📋 Detailed Results"):
        for model, results in results_by_model.items():
            st.subheader(model)
            df = pd.DataFrame([
                {
                    'Image': r['filename'],
                    'Predicted': r['vin'],
                    'Ground Truth': r.get('ground_truth', 'N/A'),
                    'Match': '✓' if r.get('exact_match') else '✗' if r.get('ground_truth') else '-',
                    'Confidence': f"{r['confidence']:.1%}",
                    'Time': f"{r['processing_time']:.2f}s"
                }
                for r in results[:50]  # Limit display
            ])
            st.dataframe(df, width="stretch")
    
    # Export option
    if st.button("📥 Export Results as CSV"):
        all_results = []
        for model, results in results_by_model.items():
            for r in results:
                all_results.append({
                    'Model': model,
                    'Image': r['filename'],
                    'Predicted VIN': r['vin'],
                    'Ground Truth': r.get('ground_truth', ''),
                    'Exact Match': r.get('exact_match', ''),
                    'Confidence': r['confidence'],
                    'Processing Time': r['processing_time']
                })
        
        df_export = pd.DataFrame(all_results)
        csv = df_export.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            "evaluation_results.csv",
            "text/csv"
        )


def process_uploaded_batch(uploaded_files):
    """Process batch of uploaded files."""
    model_name = st.session_state.current_model
    results = []
    
    progress_bar = st.progress(0)
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / len(uploaded_files))
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name
        
        result = recognize_with_model(tmp_path, model_name)
        result['filename'] = uploaded_file.name
        results.append(result)
        
        os.unlink(tmp_path)
    
    # Display results
    st.subheader("Results")
    df = pd.DataFrame([
        {
            'Image': r['filename'],
            'VIN': r['vin'],
            'Confidence': f"{r['confidence']:.1%}",
            'Time': f"{r['processing_time']:.2f}s"
        }
        for r in results
    ])
    st.dataframe(df, width="stretch")


def render_training_page():
    """Render the training/fine-tuning page."""
    st.markdown('<h1 class="main-header">🎯 Model Training</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Train or fine-tune models on your VIN dataset</p>', unsafe_allow_html=True)
    
    # Training mode selection
    st.info("""
    **Training Modes:**
    - **Fine-Tuning**: Adapt a pre-trained model to VIN data (500-5,000 images, hours)
    - **Train from Scratch**: Build a new model from random weights (50,000+ images, days)
    """)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🔧 PaddleOCR Fine-Tuning", 
        "🏗️ PaddleOCR from Scratch",
        "🤖 DeepSeek Fine-Tuning",
        "🆕 DeepSeek from Scratch"
    ])
    
    with tab1:
        render_paddleocr_finetuning()
    
    with tab2:
        render_paddleocr_scratch()
    
    with tab3:
        render_deepseek_finetuning()
    
    with tab4:
        render_deepseek_scratch()


def render_paddleocr_finetuning():
    """Render PaddleOCR fine-tuning interface."""
    st.subheader("🔧 PaddleOCR Fine-Tuning")
    st.caption("Adapt a pre-trained model to VIN data (recommended for most users)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        train_dir = st.text_input("Training Data Directory", "./dagshub_data/train", key="ft_train_dir")
        test_dir = st.text_input("Test Data Directory", "./dagshub_data/test", key="ft_test_dir")
        
        st.markdown("#### Training Parameters")
        epochs = st.slider("Number of Epochs", 1, 50, 10, key="ft_epochs")
        batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1, key="ft_batch")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.00005, 0.0001, 0.0005, 0.001],
            value=0.0005,
            key="ft_lr"
        )
    
    with col2:
        st.markdown("#### Model Configuration")
        base_model = st.selectbox(
            "Base Model",
            ["PP-OCRv5", "PP-OCRv4", "PP-OCRv3"],
            key="ft_base_model"
        )
        
        st.markdown("#### Hardware & Export")
        use_gpu = st.checkbox("Use GPU", value=True, key="ft_gpu")
        use_amp = st.checkbox("Mixed Precision (AMP)", value=True, key="ft_amp")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="ft_onnx")
        
        st.markdown("#### Output")
        output_dir = st.text_input("Output Directory", "./output/vin_rec_finetune", key="ft_output")
    
    # Training command preview
    st.markdown("#### Training Command")
    onnx_flag = " --export-onnx" if export_onnx else ""
    gpu_flag = "" if use_gpu else " --no-gpu"
    cmd = f"""python finetune_paddleocr.py \\
    --config configs/vin_finetune_config.yml \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate}{gpu_flag}{onnx_flag}"""
    
    st.code(cmd, language="bash")
    
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("🚀 Start Fine-Tuning", type="primary", key="paddle_finetune_btn"):
            st.session_state.training_status = "running"
            st.info("Fine-tuning started... Check terminal for progress.")
            st.warning("Note: Training runs in terminal. Use the command above.")
    
    with col_b:
        if st.button("📊 View Training Logs", key="paddle_logs"):
            log_path = Path(output_dir) / "train.log"
            if log_path.exists():
                with open(log_path) as f:
                    st.code(f.read()[-5000:])  # Last 5000 chars
            else:
                st.info("No training logs found yet")


def render_deepseek_finetuning():
    """Render DeepSeek fine-tuning interface."""
    st.subheader("🤖 DeepSeek-OCR Fine-Tuning")
    st.caption("Adapt the pre-trained DeepSeek-OCR model to VIN data")
    
    st.warning("⚠️ DeepSeek fine-tuning requires significant GPU memory (16GB+ with LoRA)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        train_data = st.text_input("Training Labels File", "./finetune_data/train_labels.txt", key="dsft_train")
        val_data = st.text_input("Validation Labels File", "./finetune_data/val_labels.txt", key="dsft_val")
        
        st.markdown("#### Training Parameters")
        epochs = st.slider("Number of Epochs", 1, 20, 10, key="dsft_epochs")
        batch_size = st.selectbox("Batch Size", [1, 2, 4, 8], index=1, key="dsft_batch")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00001, 0.00002, 0.00005, 0.0001],
            value=0.00002,
            key="dsft_lr"
        )
    
    with col2:
        st.markdown("#### LoRA Configuration")
        use_lora = st.checkbox("Use LoRA (recommended)", value=True, key="dsft_lora")
        if use_lora:
            lora_r = st.slider("LoRA Rank", 4, 64, 16, key="dsft_lora_r")
            lora_alpha = st.slider("LoRA Alpha", 8, 128, 32, key="dsft_lora_alpha")
            lora_dropout = st.slider("LoRA Dropout", 0.0, 0.2, 0.05, key="dsft_lora_drop")
        
        st.markdown("#### Hardware & Export")
        use_8bit = st.checkbox("8-bit Quantization (saves memory)", value=False, key="dsft_8bit")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="dsft_onnx")
        
        st.markdown("#### Output")
        output_dir = st.text_input("Output Directory", "./output/deepseek_vin_finetune", key="dsft_output")
    
    # Training command preview
    st.markdown("#### Training Command")
    onnx_flag = " --export-onnx" if export_onnx else ""
    quant_flag = " --8bit" if use_8bit else ""
    cmd = f"""python finetune_deepseek.py \\
    --config configs/deepseek_finetune_config.yml \\
    {'--lora' if use_lora else '--full'}{quant_flag}{onnx_flag}"""
    
    st.code(cmd, language="bash")
    
    if st.button("🚀 Start DeepSeek Fine-Tuning", type="primary", key="dsft_train_btn"):
        st.warning("Note: Training runs in terminal. Use the command above.")


def render_paddleocr_scratch():
    """Render PaddleOCR train from scratch interface."""
    st.subheader("🏗️ PaddleOCR Train from Scratch")
    st.caption("Build a new model from random weights (requires large dataset)")
    
    st.error("""
    ⚠️ **Training from Scratch Requirements:**
    - 50,000+ labeled VIN images
    - 24GB+ GPU memory recommended
    - Training time: Days to weeks
    - Use fine-tuning instead if you have < 10,000 images
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        train_dir = st.text_input("Training Data Directory", "./data/train", key="ps_train")
        train_labels = st.text_input("Training Labels File", "./data/train/labels.txt", key="ps_train_labels")
        val_dir = st.text_input("Validation Data Directory", "./data/val", key="ps_val")
        val_labels = st.text_input("Validation Labels File", "./data/val/labels.txt", key="ps_val_labels")
        
        st.markdown("#### Training Parameters")
        epochs = st.slider("Number of Epochs", 10, 200, 100, key="ps_epochs")
        batch_size = st.selectbox("Batch Size", [16, 32, 64, 128], index=2, key="ps_batch")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
            value=0.001,
            key="ps_lr"
        )
    
    with col2:
        st.markdown("#### Architecture")
        architecture = st.selectbox(
            "Model Architecture",
            ["SVTR_LCNet", "SVTR_Tiny", "CRNN"],
            help="SVTR_LCNet: Best accuracy. CRNN: Classic, faster training.",
            key="ps_arch"
        )
        
        backbone = st.selectbox(
            "Backbone",
            ["PPLCNetV3", "MobileNetV3", "ResNet"],
            key="ps_backbone"
        )
        
        st.markdown("#### Hardware & Export")
        use_gpu = st.checkbox("Use GPU", value=True, key="ps_gpu")
        use_amp = st.checkbox("Mixed Precision (AMP)", value=True, key="ps_amp")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="ps_onnx")
        
        st.markdown("#### Output")
        output_dir = st.text_input("Output Directory", "./output/vin_scratch_train", key="ps_output")
    
    # Training command preview
    st.markdown("#### Training Command")
    onnx_flag = " --export-onnx" if export_onnx else ""
    cmd = f"""python train_from_scratch.py --model paddleocr \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate} \\
    --output-dir {output_dir}{onnx_flag}"""
    
    st.code(cmd, language="bash")
    
    if st.button("🏗️ Start Training from Scratch", type="primary", key="ps_train_btn"):
        st.warning("⚠️ This will take a LONG time. Use the command in terminal for monitoring.")


def render_deepseek_scratch():
    """Render DeepSeek train from scratch interface."""
    st.subheader("🆕 Vision-Language Model from Scratch")
    st.caption("Build a custom vision-language model for VIN recognition")
    
    st.error("""
    ⚠️ **Training from Scratch Requirements:**
    - 100,000+ labeled images recommended
    - 48GB+ GPU memory (A100 or better)
    - Training time: Weeks
    - This creates a SMALLER model than DeepSeek-OCR
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Dataset Configuration")
        train_data = st.text_input("Training Labels File", "./data/train/labels.txt", key="dss_train")
        val_data = st.text_input("Validation Labels File", "./data/val/labels.txt", key="dss_val")
        data_dir = st.text_input("Data Directory", "./data", key="dss_data_dir")
        
        st.markdown("#### Training Parameters")
        epochs = st.slider("Number of Epochs", 10, 100, 50, key="dss_epochs")
        batch_size = st.selectbox("Batch Size", [2, 4, 8, 16], index=1, key="dss_batch")
        grad_accum = st.slider("Gradient Accumulation Steps", 1, 16, 8, key="dss_grad_accum")
        learning_rate = st.select_slider(
            "Learning Rate",
            options=[0.00005, 0.0001, 0.0002, 0.0005],
            value=0.0001,
            key="dss_lr"
        )
    
    with col2:
        st.markdown("#### Model Configuration")
        image_size = st.selectbox("Image Size", [256, 384, 512], index=1, key="dss_img_size")
        
        st.markdown("#### Hardware & Export")
        use_bf16 = st.checkbox("BFloat16 (recommended for Ampere+ GPUs)", value=True, key="dss_bf16")
        use_fp16 = st.checkbox("Float16 (for older GPUs)", value=False, key="dss_fp16")
        export_onnx = st.checkbox("Export to ONNX after training", value=False, key="dss_onnx")
        
        st.markdown("#### Output")
        output_dir = st.text_input("Output Directory", "./output/deepseek_scratch_train", key="dss_output")
    
    # Training command preview
    st.markdown("#### Training Command")
    onnx_flag = " --export-onnx" if export_onnx else ""
    cmd = f"""python train_from_scratch.py --model deepseek \\
    --epochs {epochs} \\
    --batch-size {batch_size} \\
    --lr {learning_rate} \\
    --output-dir {output_dir}{onnx_flag}"""
    
    st.code(cmd, language="bash")
    
    if st.button("🆕 Start Training from Scratch", type="primary", key="dss_train_btn"):
        st.warning("⚠️ This will take a VERY LONG time. Use the command in terminal for monitoring.")


def render_results_dashboard():
    """Render the results dashboard page."""
    st.markdown('<h1 class="main-header">📈 Results Dashboard</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">View and analyze recognition results</p>', unsafe_allow_html=True)
    
    # Check for saved results
    results_dir = Path("./results")
    
    if results_dir.exists():
        result_files = list(results_dir.glob("*.json")) + list(results_dir.glob("*.csv"))
        
        if result_files:
            st.subheader("📁 Saved Results")
            
            selected_file = st.selectbox(
                "Select result file",
                [f.name for f in result_files]
            )
            
            if selected_file:
                file_path = results_dir / selected_file
                
                if selected_file.endswith('.json'):
                    with open(file_path) as f:
                        data = json.load(f)
                    st.json(data)
                else:
                    df = pd.read_csv(file_path)
                    st.dataframe(df, width="stretch")
                    
                    # Quick charts
                    if PLOTLY_AVAILABLE and 'F1 Micro' in df.columns:
                        fig = px.bar(df, x='Model', y='F1 Micro', title='F1 Micro Score by Model')
                        st.plotly_chart(fig, width="stretch")
    
    # Session history
    st.subheader("📊 Session History")
    
    if st.session_state.results_history:
        df = pd.DataFrame([
            {
                'Time': r.get('timestamp', 'N/A')[:19],
                'Model': r.get('model', 'N/A'),
                'Image': r.get('filename', 'N/A'),
                'VIN': r.get('vin', 'N/A'),
                'Confidence': f"{r.get('confidence', 0):.1%}",
            }
            for r in st.session_state.results_history[-50:]  # Last 50
        ])
        st.dataframe(df, width="stretch")
        
        if st.button("🗑️ Clear History"):
            st.session_state.results_history = []
            st.rerun()
    else:
        st.info("No recognition results in this session yet")


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get selected page
    page = render_sidebar()
    
    # Render selected page
    if page == "🔍 Recognition":
        render_recognition_page()
    elif page == "📊 Batch Evaluation":
        render_batch_evaluation_page()
    elif page == "🎯 Training":
        render_training_page()
    elif page == "📈 Results Dashboard":
        render_results_dashboard()


if __name__ == "__main__":
    main()
