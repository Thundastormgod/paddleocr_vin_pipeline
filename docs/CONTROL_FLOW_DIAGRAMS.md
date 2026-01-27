# VIN OCR Training: Control Flow Diagrams (Mermaid)

This document contains Mermaid diagrams for visualizing the training pipeline control flow.

## 1. System Architecture

```mermaid
graph TB
    subgraph "Entry Points"
        A[train_pipeline.py] 
        B[finetune_paddleocr.py]
    end
    
    subgraph "Rule-Based Path"
        C[VINTrainingPipeline]
        D[train_rule_learning]
        E[VINOCRPipeline.recognize]
        F[_build_correction_rules]
        G[model.json]
    end
    
    subgraph "Neural Path"
        H[VINFineTuner]
        I[VINRecognitionModel]
        J[VINRecognitionDataset]
        K[CTC Loss Training]
        L[best_accuracy.pdparams]
    end
    
    subgraph "Shared Resources"
        M[vin_utils.py]
        N[configs/vin_finetune_config.yml]
    end
    
    A -->|method=rules| C
    A -->|method=finetune| C
    C -->|rules| D
    C -->|finetune| B
    D --> E
    E --> F
    F --> G
    
    B --> H
    H --> I
    H --> J
    I --> K
    K --> L
    
    C -.->|imports| M
    H -.->|loads| N
```

## 2. Rule-Based Learning Flow

```mermaid
flowchart TD
    A[Start: train_rule_learning] --> B[Initialize OCR Pipeline]
    B --> C[Load Training Dataset]
    C --> D[Load Validation Dataset]
    
    D --> E{For each training image}
    E --> F[Run OCR Recognition]
    F --> G[Compare prediction vs ground truth]
    G --> H[Store result]
    H --> E
    
    E -->|Complete| I[Calculate Baseline Accuracy]
    I --> J[Build Confusion Matrix]
    J --> K[Extract Correction Rules]
    K --> L[Save model.json]
    
    L --> M{Validation data exists?}
    M -->|Yes| N[Evaluate with rules]
    M -->|No| O[Return Results]
    N --> O
    
    subgraph "Rule Extraction"
        J1[For each misprediction]
        J2[Count char confusions]
        J3[Select most frequent correction]
        J1 --> J2 --> J3
    end
    
    J -.-> J1
```

## 3. Neural Fine-Tuning Flow

```mermaid
flowchart TD
    A[Start: VINFineTuner.train] --> B{Resume checkpoint?}
    B -->|Yes| C[Load checkpoint]
    B -->|No| D[Start fresh]
    C --> D
    
    D --> E[For epoch = 1 to N]
    
    subgraph "Training Epoch"
        E1[model.train]
        E2[For each batch]
        E3[Forward Pass]
        E4[CTC Loss]
        E5[Backward Pass]
        E6[Optimizer Step]
        E1 --> E2 --> E3 --> E4 --> E5 --> E6
        E6 -->|More batches| E2
    end
    
    E --> E1
    E6 -->|Epoch done| F[lr_scheduler.step]
    
    subgraph "Validation"
        V1[model.eval]
        V2[For each val batch]
        V3[Forward Pass]
        V4[Decode predictions]
        V5[Calculate accuracy]
        V1 --> V2 --> V3 --> V4 --> V5
        V5 -->|More batches| V2
    end
    
    F --> V1
    V5 -->|Done| G{accuracy > best?}
    G -->|Yes| H[Update best_accuracy]
    G -->|No| I{Save checkpoint?}
    H --> I
    I -->|Yes| J[save_checkpoint]
    I -->|No| K{More epochs?}
    J --> K
    K -->|Yes| E
    K -->|No| L[export_inference_model]
    L --> M[End]
```

## 4. Data Processing Pipeline

```mermaid
flowchart LR
    subgraph "Input"
        I1[Raw Image<br/>H×W×3]
        I2[VIN Label<br/>'ABC123...']
    end
    
    subgraph "Image Processing"
        P1[Resize<br/>48×320]
        P2[Normalize<br/>[-1,1]]
        P3[Augment<br/>optional]
        P4[HWC→CHW]
    end
    
    subgraph "Label Processing"
        L1[Char to Index<br/>Encoding]
        L2[Pad to<br/>max_length]
    end
    
    subgraph "DataLoader"
        D1[Batch<br/>images]
        D2[Batch<br/>labels]
        D3[lengths]
    end
    
    I1 --> P1 --> P2 --> P3 --> P4 --> D1
    I2 --> L1 --> L2 --> D2
    L2 --> D3
```

## 5. Model Architecture

```mermaid
flowchart TD
    subgraph "VINRecognitionModel"
        subgraph "Backbone (PPLCNet-like)"
            B1[Conv2D 3→32, s=2]
            B2[Conv2D 32→64, s=2]
            B3[Conv2D 64→128, s=2,1]
            B4[Conv2D 128→256, s=2,1]
            B5[Conv2D 256→512, s=2,1]
            B1 --> B2 --> B3 --> B4 --> B5
        end
        
        subgraph "Neck"
            N1[AdaptiveAvgPool2D<br/>1,None]
            N2[Flatten]
            N3[Transpose]
            N1 --> N2 --> N3
        end
        
        subgraph "Head (CTC)"
            H1[Linear 512→256]
            H2[ReLU + Dropout]
            H3[Linear 256→num_classes]
            H1 --> H2 --> H3
        end
        
        B5 --> N1
        N3 --> H1
    end
    
    Input[Input<br/>B×3×48×320] --> B1
    H3 --> Output[Output<br/>B×T×C]
```

## 6. CTC Loss Computation

```mermaid
flowchart TD
    A[Model Output<br/>logits: B×T×C] --> B[log_softmax]
    B --> C[Transpose<br/>T×B×C]
    
    D[Target Labels<br/>B×17] --> E[Label Lengths<br/>B]
    
    F[Input Lengths<br/>B filled with T]
    
    C --> G[CTC Loss]
    E --> G
    F --> G
    D --> G
    
    G --> H[Scalar Loss]
    H --> I[Backward<br/>∂L/∂θ]
    I --> J[Optimizer Update<br/>θ ← θ - lr·∇θ]
    
    subgraph "CTC Alignment"
        G1[All valid paths π]
        G2[Sum probabilities]
        G3[-log likelihood]
        G1 --> G2 --> G3
    end
```

## 7. Class Dependency Graph

```mermaid
classDiagram
    class VINTrainingPipeline {
        +dataset_dir: Path
        +output_dir: Path
        +batch_size: int
        +epochs: int
        +learning_rate: float
        +use_gpu: bool
        +pipeline: VINOCRPipeline
        +train(method, verbose)
        +train_rule_learning(verbose)
        +_full_finetuning()
        +load_dataset(split)
        +augment_image(image)
    }
    
    class VINFineTuner {
        +config: Dict
        +output_dir: Path
        +model: VINRecognitionModel
        +optimizer: Adam
        +criterion: CTCLoss
        +train_loader: DataLoader
        +val_loader: DataLoader
        +train(resume_from)
        +train_epoch(epoch)
        +validate()
        +save_checkpoint()
        +export_inference_model()
    }
    
    class VINRecognitionModel {
        +backbone: Sequential
        +neck: Sequential
        +head: Sequential
        +forward(x)
    }
    
    class VINRecognitionDataset {
        +data_dir: Path
        +char_dict: Dict
        +samples: List
        +__getitem__(idx)
        +__len__()
    }
    
    class VINOCRPipeline {
        +recognize(image_path)
    }
    
    VINTrainingPipeline --> VINOCRPipeline : uses
    VINTrainingPipeline ..> VINFineTuner : spawns
    VINFineTuner --> VINRecognitionModel : trains
    VINFineTuner --> VINRecognitionDataset : loads
    VINRecognitionDataset --|> paddle.io.Dataset
    VINRecognitionModel --|> paddle.nn.Layer
```

## 8. Training Method Decision Tree

```mermaid
flowchart TD
    A[Dataset Size?] -->|< 500 images| B[Use Rule-Based]
    A -->|500-2000| C[Use Rule-Based +<br/>Data Augmentation]
    A -->|> 2000| D[Use Neural Fine-Tuning]
    
    B --> E[Fast, No GPU needed]
    C --> F[Better coverage,<br/>Still fast]
    D --> G{GPU Available?}
    
    G -->|Yes| H[Full Fine-tuning<br/>with AMP]
    G -->|No| I[CPU Training<br/>Slower but works]
    
    H --> J[Best Results]
    I --> K[Good Results]
    E --> L[Quick Improvement]
    F --> L
```

## 9. File I/O Dependencies

```mermaid
flowchart LR
    subgraph "Input Files"
        IF1[dataset/train_labels.txt]
        IF2[dataset/val_labels.txt]
        IF3[dataset/images/]
        IF4[configs/vin_finetune_config.yml]
        IF5[configs/vin_dict.txt]
    end
    
    subgraph "Processing"
        P1[train_pipeline.py]
        P2[finetune_paddleocr.py]
    end
    
    subgraph "Output Files"
        OF1[training_output/checkpoints/model.json]
        OF2[training_output/model/best_accuracy.pdparams]
        OF3[training_output/model/latest.pdparams]
        OF4[training_output/model/inference/]
        OF5[training_output/runtime_config.yml]
    end
    
    IF1 --> P1
    IF2 --> P1
    IF3 --> P1
    IF4 --> P2
    IF5 --> P2
    
    P1 -->|rule-based| OF1
    P1 -->|finetune| P2
    P2 --> OF2
    P2 --> OF3
    P2 --> OF4
    P1 --> OF5
```

## 10. State Machine: Training Lifecycle

```mermaid
stateDiagram-v2
    [*] --> Initialized: __init__
    
    Initialized --> ValidatingData: _validate_dataset
    ValidatingData --> DataValid: success
    ValidatingData --> Error: missing files
    
    DataValid --> LoadingPaddle: _load_paddle
    LoadingPaddle --> PaddleReady: success
    LoadingPaddle --> CPUOnly: no GPU
    
    PaddleReady --> RuleBased: method=rules
    CPUOnly --> RuleBased: method=rules
    
    PaddleReady --> NeuralTraining: method=finetune
    CPUOnly --> NeuralTraining: method=finetune
    
    state RuleBased {
        [*] --> CollectingPredictions
        CollectingPredictions --> BuildingRules
        BuildingRules --> Evaluating
        Evaluating --> [*]
    }
    
    state NeuralTraining {
        [*] --> PrepareData
        PrepareData --> SpawnProcess
        SpawnProcess --> TrainingLoop
        TrainingLoop --> ExportModel
        ExportModel --> [*]
    }
    
    RuleBased --> Complete
    NeuralTraining --> Complete
    
    Complete --> [*]
    Error --> [*]
```

---

## Usage

To render these diagrams:
1. Use a Markdown viewer with Mermaid support (VS Code with Mermaid extension, GitHub, etc.)
2. Paste into [Mermaid Live Editor](https://mermaid.live/)
3. Export as PNG/SVG for documentation

## Quick Reference

| Diagram | Purpose |
|---------|---------|
| System Architecture | High-level component relationships |
| Rule-Based Flow | Step-by-step rule learning process |
| Neural Fine-Tuning Flow | Complete training loop |
| Data Processing | Image and label preprocessing |
| Model Architecture | Neural network structure |
| CTC Loss | Loss computation details |
| Class Dependency | OOP relationships |
| Decision Tree | When to use which method |
| File I/O | Input/output file mapping |
| State Machine | Training lifecycle states |
