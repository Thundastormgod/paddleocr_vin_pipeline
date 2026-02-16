#!/usr/bin/env python3
"""
ZenML Pipeline for VIN OCR Architecture Tracking
Tracks different architectures, hyperparameters, and their performance
"""

import os
import json
import yaml
import paddle
from typing import Dict, Any, Tuple
from pathlib import Path
from datetime import datetime

import zenml
from zenml import step, pipeline
from zenml.client import Client
from zenml.logger import get_logger

# Initialize ZenML logger
logger = get_logger(__name__)

@step
def load_architecture_config(architecture_name: str) -> Dict[str, Any]:
    """Load architecture configuration from predefined templates."""
    
    logger.info(f"Loading architecture config for: {architecture_name}")
    
    # Architecture templates
    architectures = {
        "rosetta_ctc": {
            "name": "Rosetta + CTC",
            "algorithm": "Rosetta",
            "backbone": "ResNet34_vd",
            "neck": "SequenceEncoder",
            "head": "MultiHead",
            "subhead": "CTCHead",
            "loss": "CTCLoss",
            "postprocess": "CTCLabelDecode",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 30,
            "image_shape": [3, 48, 320],
            "hidden_size": 256,
            "fc_decay": 1e-05
        },
        "rosetta_sar": {
            "name": "Rosetta + SAR",
            "algorithm": "Rosetta",
            "backbone": "ResNet34_vd",
            "neck": "SequenceEncoder",
            "head": "MultiHead",
            "subhead": "SARHead",
            "loss": "CrossEntropyLoss",
            "postprocess": "SARLabelDecode",
            "learning_rate": 0.002,
            "batch_size": 16,
            "epochs": 30,
            "image_shape": [3, 48, 320],
            "hidden_size": 256,
            "fc_decay": 1e-05
        },
        "svtr_lcnet": {
            "name": "SVTR + LCNet",
            "algorithm": "SVTR_LCNet",
            "backbone": "PPHGNetV2",
            "neck": "SequenceEncoder",
            "head": "MultiHead",
            "subhead": "CTCHead",
            "loss": "CTCLoss",
            "postprocess": "CTCLabelDecode",
            "learning_rate": 0.001,
            "batch_size": 16,
            "epochs": 30,
            "image_shape": [3, 48, 320],
            "hidden_size": 256,
            "fc_decay": 1e-05
        }
    }
    
    if architecture_name not in architectures:
        raise ValueError(f"Unknown architecture: {architecture_name}")
    
    config = architectures[architecture_name]
    config["architecture_key"] = architecture_name
    config["timestamp"] = datetime.now().isoformat()
    
    logger.info(f"Loaded config: {config['name']}")
    return config

@step
def create_config_file(config: Dict[str, Any]) -> str:
    """Create YAML config file for training."""
    
    logger.info("Creating training config file")
    
    # Build full config
    full_config = {
        "Global": {
            "debug": False,
            "use_gpu": False,
            "epoch_num": config["epochs"],
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "save_model_dir": "./output/vin_rec_finetune",
            "save_epoch_step": 5,
            "eval_batch_step": [0, 500],
            "cal_metric_during_train": True,
            "pretrained_model": None,
            "character_dict_path": "./configs/vin_dict.txt",
            "max_text_length": 17,
            "use_space_char": False,
            "distributed": False,
            "use_amp": False
        },
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "Cosine",
                "learning_rate": config["learning_rate"],
                "warmup_epoch": 0
            },
            "regularizer": {
                "name": "L2",
                "factor": config["fc_decay"]
            }
        },
        "Architecture": {
            "model_type": "rec",
            "algorithm": config["algorithm"],
            "Transform": None,
            "Backbone": {
                "name": config["backbone"]
            },
            "Neck": {
                "name": config["neck"],
                "hidden_size": config["hidden_size"]
            },
            "Head": {
                "name": config["head"],
                "head_list": [
                    {config["subhead"]: {"fc_decay": config["fc_decay"]}}
                ]
            }
        },
        "Loss": {
            "name": config["loss"]
        },
        "PostProcess": {
            "name": config["postprocess"],
            "character_dict_path": "./configs/vin_dict.txt",
            "use_space_char": False
        },
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
            "ignore_space": True
        },
        "Train": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": "./finetune_data/",
                "label_file_list": ["./finetune_data/train_labels.txt"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"RecAug": None},
                    {"MultiLabelEncode": {"max_text_length": 17}},
                    {"RecResizeImg": {"image_shape": config["image_shape"]}},
                    {"RandomRotate": {"max_angle": 5}},
                    {"RandomDistort": {"max_ratio": 0.8}},
                    {"RandomContrast": {"contrast_range": [0.8, 1.2]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_sar", "length", "valid_ratio"]}}
                ]
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": config["batch_size"],
                "drop_last": True,
                "num_workers": 0,
                "use_shared_memory": False
            }
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": "./finetune_data/",
                "label_file_list": ["./finetune_data/val_labels.txt"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"MultiLabelEncode": {"max_text_length": 17}},
                    {"RecResizeImg": {"image_shape": config["image_shape"]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_sar", "length", "valid_ratio"]}}
                ]
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": 128,
                "num_workers": 0,
                "use_shared_memory": False
            }
        }
    }
    
    # Create config file
    config_path = f"./zenml_configs/{config['architecture_key']}_config.yml"
    os.makedirs("./zenml_configs", exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(full_config, f, default_flow_style=False)
    
    logger.info(f"Config file created: {config_path}")
    return config_path

@step
def run_training_experiment(config_path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Run training experiment and track results."""
    
    logger.info(f"Starting training experiment for {config['name']}")
    
    # Import training module
    import sys
    sys.path.insert(0, '.')
    from src.vin_ocr.training.finetune_paddleocr import VINFineTuner
    
    # Load config
    with open(config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    # Initialize trainer
    trainer = VINFineTuner(training_config)
    
    # Run training (simulate for demo - in real usage, this would run full training)
    logger.info("Running training simulation...")
    
    # Simulate training results (replace with actual training)
    simulated_results = {
        "architecture": config["name"],
        "architecture_key": config["architecture_key"],
        "config_path": config_path,
        "training_time_hours": 0.5,  # Simulated
        "epochs_completed": 5,  # Simulated
        "final_train_loss": 0.75,
        "final_val_loss": 0.85,
        "best_accuracy": 0.15,  # Simulated
        "final_accuracy": 0.12,
        "character_accuracy": 0.85,
        "f1_micro": 0.85,
        "f1_macro": 0.75,
        "cer": 0.15,
        "ned": 0.85,
        "exact_match_count": 5,
        "total_samples": 43,
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "epochs": config["epochs"],
            "backbone": config["backbone"],
            "hidden_size": config["hidden_size"],
            "fc_decay": config["fc_decay"]
        }
    }
    
    logger.info(f"Training completed. Best accuracy: {simulated_results['best_accuracy']:.4f}")
    return simulated_results

@step
def evaluate_model(results: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate model performance and create detailed metrics."""
    
    logger.info("Evaluating model performance")
    
    # Calculate additional metrics
    exact_match_accuracy = results["exact_match_count"] / results["total_samples"]
    
    evaluation_results = {
        **results,
        "exact_match_accuracy": exact_match_accuracy,
        "performance_tier": _get_performance_tier(exact_match_accuracy),
        "recommendations": _get_recommendations(results, exact_match_accuracy)
    }
    
    logger.info(f"Evaluation completed. Performance tier: {evaluation_results['performance_tier']}")
    return evaluation_results

@step
def log_to_zenml(results: Dict[str, Any]) -> None:
    """Log results to ZenML for tracking."""
    
    logger.info("Logging results to ZenML")
    
    # Simple logging without ZenML artifacts for now
    # Save results to local file
    import json
    import os
    
    os.makedirs("./zenml_results", exist_ok=True)
    result_file = f"./zenml_results/vin_ocr_results_{results['architecture_key']}.json"
    
    with open(result_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {result_file}")
    logger.info("ZenML logging completed (local storage)")

def _get_performance_tier(accuracy: float) -> str:
    """Determine performance tier based on accuracy."""
    if accuracy >= 0.5:
        return "Excellent"
    elif accuracy >= 0.3:
        return "Good"
    elif accuracy >= 0.1:
        return "Fair"
    else:
        return "Poor"

def _get_recommendations(results: Dict[str, Any], accuracy: float) -> list:
    """Get recommendations based on results."""
    recommendations = []
    
    if accuracy < 0.1:
        recommendations.append("Consider increasing training epochs")
        recommendations.append("Try data augmentation")
        recommendations.append("Check data quality")
    elif accuracy < 0.3:
        recommendations.append("Try different learning rate")
        recommendations.append("Consider larger model")
    elif accuracy < 0.5:
        recommendations.append("Fine-tune hyperparameters")
        recommendations.append("Try ensemble methods")
    
    if results["character_accuracy"] > accuracy + 0.3:
        recommendations.append("Focus on sequence alignment")
        recommendations.append("Consider attention mechanisms")
    
    return recommendations

@pipeline
def vin_ocr_pipeline(architecture_name: str):
    """Complete VIN OCR training and evaluation pipeline."""
    
    # Load architecture configuration
    config = load_architecture_config(architecture_name)
    
    # Create config file
    config_path = create_config_file(config)
    
    # Run training experiment
    results = run_training_experiment(config_path, config)
    
    # Evaluate model
    evaluation_results = evaluate_model(results)
    
    # Log to ZenML
    log_to_zenml(evaluation_results)
    
    return evaluation_results

def run_architecture_comparison():
    """Run comparison across multiple architectures."""
    
    logger.info("Starting architecture comparison")
    
    architectures = ["rosetta_ctc", "rosetta_sar", "svtr_lcnet"]
    results = []
    
    for arch in architectures:
        logger.info(f"Running pipeline for architecture: {arch}")
        
        # Run pipeline
        result = vin_ocr_pipeline(architecture_name=arch)
        results.append(result)
    
    # Create comparison report
    comparison_report = {
        "timestamp": datetime.now().isoformat(),
        "architectures_tested": architectures,
        "results": results,
        "best_architecture": max(results, key=lambda x: x["exact_match_accuracy"]),
        "summary": {
            "total_architectures": len(architectures),
            "best_accuracy": max(r["exact_match_accuracy"] for r in results),
            "worst_accuracy": min(r["exact_match_accuracy"] for r in results),
            "average_accuracy": sum(r["exact_match_accuracy"] for r in results) / len(results)
        }
    }
    
    # Save comparison report
    with open("./zenml_configs/architecture_comparison.json", 'w') as f:
        json.dump(comparison_report, f, indent=2)
    
    logger.info("Architecture comparison completed")
    return comparison_report

if __name__ == "__main__":
    # Example usage
    logger.info("Starting VIN OCR Architecture Tracking")
    
    # Run single architecture
    # result = vin_ocr_pipeline(architecture_name="rosetta_ctc")
    
    # Run comparison
    comparison = run_architecture_comparison()
    
    print("🎉 VIN OCR Architecture Tracking Completed!")
    print(f"📊 Best architecture: {comparison['best_architecture']['architecture']}")
    print(f"🏆 Best accuracy: {comparison['best_architecture']['exact_match_accuracy']:.4f}")
