# Neural Fine-Tuning Techniques for VIN OCR

## Recommended Techniques for This Implementation

Based on the VIN OCR use case (17-character fixed-length alphanumeric recognition), here are the **optimal fine-tuning techniques** ranked by effectiveness:

---

## 1. **Transfer Learning with Frozen Backbone** ⭐ Recommended First

### Why for VIN?
- VIN characters are standard alphanumeric (no special fonts)
- Pretrained PP-OCRv4 already knows character shapes
- Only need to adapt the head for VIN-specific patterns

### Implementation

```python
# Freeze backbone, train only head
def freeze_backbone(model):
    """Freeze backbone weights, train only recognition head."""
    for name, param in model.named_parameters():
        if 'backbone' in name.lower():
            param.stop_gradient = True  # PaddlePaddle syntax
            # param.requires_grad = False  # PyTorch syntax
        else:
            param.stop_gradient = False
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

# Usage
model = load_pretrained_model("PP-OCRv4_rec")
freeze_backbone(model)
# Train for 20-50 epochs with lr=0.001
```

### Config Changes
```yaml
Optimizer:
  lr:
    learning_rate: 0.001  # Higher LR since only training head
    warmup_epoch: 2
Global:
  epoch_num: 50  # Fewer epochs needed
```

---

## 2. **Discriminative Learning Rates** ⭐⭐ Best for Production

### Why for VIN?
- Different layers learn different features
- Lower layers: generic edge/texture features (already good)
- Higher layers: character-specific features (need more adaptation)

### Implementation

```python
def get_discriminative_lr_params(model, base_lr=0.0001, lr_mult=0.1):
    """
    Apply different learning rates to different layers.
    
    Layer-wise LR schedule:
    - Backbone early layers: base_lr × 0.01 (almost frozen)
    - Backbone late layers: base_lr × 0.1
    - Neck: base_lr × 0.5
    - Head: base_lr × 1.0 (full learning rate)
    """
    param_groups = []
    
    for name, param in model.named_parameters():
        if param.stop_gradient:
            continue
            
        if 'backbone' in name:
            if any(x in name for x in ['conv1', 'layer1', 'layer2']):
                # Early backbone layers - minimal learning
                lr = base_lr * 0.01
            else:
                # Late backbone layers - moderate learning
                lr = base_lr * 0.1
        elif 'neck' in name:
            lr = base_lr * 0.5
        else:  # head
            lr = base_lr * 1.0
        
        param_groups.append({
            'params': [param],
            'learning_rate': lr,
            'name': name
        })
    
    return param_groups

# Usage
param_groups = get_discriminative_lr_params(model, base_lr=0.0005)
optimizer = paddle.optimizer.Adam(parameters=param_groups)
```

---

## 3. **Gradual Unfreezing** ⭐⭐ Best for Limited Data

### Why for VIN?
- Prevents catastrophic forgetting of pretrained features
- Allows model to adapt gradually
- Works well with 1000-5000 training images

### Implementation

```python
class GradualUnfreezer:
    """
    Gradually unfreeze layers during training.
    
    Schedule:
    - Epochs 1-10: Only head trainable
    - Epochs 11-20: Unfreeze neck
    - Epochs 21-30: Unfreeze late backbone
    - Epochs 31+: Full model trainable
    """
    
    def __init__(self, model, unfreeze_schedule=None):
        self.model = model
        self.schedule = unfreeze_schedule or {
            10: ['head'],
            20: ['neck'],
            30: ['backbone.layer4', 'backbone.layer3'],
            40: ['backbone'],  # Full unfreeze
        }
        self._freeze_all_except(['head'])
    
    def _freeze_all_except(self, trainable_parts):
        """Freeze all except specified parts."""
        for name, param in self.model.named_parameters():
            should_train = any(part in name for part in trainable_parts)
            param.stop_gradient = not should_train
    
    def step(self, epoch):
        """Call at start of each epoch to potentially unfreeze layers."""
        for unfreeze_epoch, parts in sorted(self.schedule.items()):
            if epoch >= unfreeze_epoch:
                current_trainable = parts
        
        self._freeze_all_except(current_trainable)
        trainable = sum(1 for p in self.model.parameters() if not p.stop_gradient)
        print(f"Epoch {epoch}: {trainable} trainable parameters")

# Usage in training loop
unfreezer = GradualUnfreezer(model)
for epoch in range(100):
    unfreezer.step(epoch)
    train_one_epoch(model, train_loader, optimizer)
```

---

## 4. **Mixed Precision Training (AMP)** ⭐ Always Use

### Why for VIN?
- 2x faster training with minimal accuracy loss
- Reduces GPU memory usage
- Essential for larger batch sizes

### Implementation (Already in your code)

```python
from paddle.amp import auto_cast, GradScaler

# Initialize scaler
scaler = GradScaler()

# Training loop
for batch in train_loader:
    # Forward pass with mixed precision
    with auto_cast():
        logits = model(images)
        loss = criterion(logits, labels)
    
    # Backward pass with scaling
    scaled_loss = scaler.scale(loss)
    scaled_loss.backward()
    
    # Optimizer step with unscaling
    scaler.step(optimizer)
    scaler.update()
    optimizer.clear_grad()
```

### Config
```yaml
Global:
  use_amp: true
  amp_level: O1  # O1 = mixed precision, O2 = pure fp16
```

---

## 5. **Label Smoothing** ⭐ Improves Generalization

### Why for VIN?
- VIN characters can be ambiguous (8 vs B, 0 vs O)
- Prevents overconfident predictions
- Improves generalization to new VIN styles

### Implementation

```python
class LabelSmoothingCTCLoss(nn.Layer):
    """
    CTC Loss with label smoothing.
    
    Instead of hard labels [0, 0, 1, 0, 0], use:
    [0.02, 0.02, 0.92, 0.02, 0.02]
    """
    
    def __init__(self, smoothing=0.1, blank=0, num_classes=37):
        super().__init__()
        self.smoothing = smoothing
        self.blank = blank
        self.num_classes = num_classes
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction='none')
    
    def forward(self, log_probs, targets, input_lengths, target_lengths):
        # Standard CTC loss
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # Add KL divergence term for smoothing
        # This encourages the model to be less confident
        uniform = paddle.full_like(log_probs, 1.0 / self.num_classes)
        kl_div = F.kl_div(log_probs, uniform, reduction='batchmean')
        
        # Combine losses
        loss = (1 - self.smoothing) * ctc_loss.mean() + self.smoothing * kl_div
        
        return loss

# Usage
criterion = LabelSmoothingCTCLoss(smoothing=0.1, num_classes=37)
```

---

## 6. **Data Augmentation Techniques** ⭐⭐ Critical for VIN

### VIN-Specific Augmentations

```python
class VINAugmentation:
    """
    Augmentation pipeline optimized for VIN images.
    
    VIN-specific considerations:
    - Preserve character shapes (no excessive distortion)
    - Simulate real-world conditions (lighting, blur, angle)
    - No horizontal flip (changes VIN meaning)
    """
    
    def __init__(self, prob=0.5):
        self.prob = prob
    
    def __call__(self, image):
        if np.random.random() > self.prob:
            return image
        
        # 1. Brightness/Contrast (common in VIN photos)
        if np.random.random() > 0.3:
            alpha = np.random.uniform(0.7, 1.3)  # Contrast
            beta = np.random.randint(-30, 30)    # Brightness
            image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        
        # 2. Small rotation (VIN plates are rarely perfectly aligned)
        if np.random.random() > 0.5:
            angle = np.random.uniform(-5, 5)  # Max 5 degrees
            h, w = image.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            image = cv2.warpAffine(image, M, (w, h), 
                                   borderMode=cv2.BORDER_REPLICATE)
        
        # 3. Gaussian blur (simulate focus issues)
        if np.random.random() > 0.7:
            ksize = np.random.choice([3, 5])
            image = cv2.GaussianBlur(image, (ksize, ksize), 0)
        
        # 4. Gaussian noise (simulate camera sensor noise)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 10, image.shape).astype(np.uint8)
            image = cv2.add(image, noise)
        
        # 5. Perspective transform (simulate viewing angle)
        if np.random.random() > 0.6:
            image = self._random_perspective(image, max_warp=0.05)
        
        # 6. JPEG compression artifacts (common in photos)
        if np.random.random() > 0.7:
            quality = np.random.randint(50, 95)
            _, encoded = cv2.imencode('.jpg', image, 
                                      [cv2.IMWRITE_JPEG_QUALITY, quality])
            image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        
        return image
    
    def _random_perspective(self, image, max_warp=0.05):
        """Apply slight perspective warp."""
        h, w = image.shape[:2]
        
        # Random corner offsets
        pts1 = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        offset = max_warp * min(w, h)
        pts2 = pts1 + np.random.uniform(-offset, offset, pts1.shape).astype(np.float32)
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        return cv2.warpPerspective(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
```

### Config for Augmentation
```yaml
Train:
  dataset:
    transforms:
      - DecodeImage:
          img_mode: BGR
      - RecConAug:  # Contextual augmentation
          prob: 0.5
          ext_data_num: 2
      - RecAug:     # Standard augmentation
      - VINAugmentation:  # Custom VIN augmentation
          prob: 0.5
```

---

## 7. **Knowledge Distillation** (For Deployment)

### Why for VIN?
- Compress large model to smaller one for edge deployment
- Maintain accuracy while reducing inference time
- Teacher: PP-OCRv4 full model → Student: PP-OCRv4 mobile

### Implementation

```python
class DistillationLoss(nn.Layer):
    """
    Knowledge distillation loss.
    
    L = α * L_hard + (1-α) * L_soft
    
    Where:
    - L_hard: Standard CTC loss with true labels
    - L_soft: KL divergence between student and teacher logits
    """
    
    def __init__(self, teacher_model, temperature=4.0, alpha=0.5):
        super().__init__()
        self.teacher = teacher_model
        self.teacher.eval()  # Teacher is always in eval mode
        
        for param in self.teacher.parameters():
            param.stop_gradient = True  # Freeze teacher
        
        self.temperature = temperature
        self.alpha = alpha
        self.ctc_loss = nn.CTCLoss(blank=0)
    
    def forward(self, student_logits, images, labels, input_lengths, target_lengths):
        # Hard loss (standard CTC)
        student_log_probs = F.log_softmax(student_logits, axis=-1)
        hard_loss = self.ctc_loss(
            student_log_probs.transpose([1, 0, 2]),
            labels, input_lengths, target_lengths
        )
        
        # Soft loss (distillation from teacher)
        with paddle.no_grad():
            teacher_logits = self.teacher(images)
        
        # Softened probabilities
        teacher_soft = F.softmax(teacher_logits / self.temperature, axis=-1)
        student_soft = F.log_softmax(student_logits / self.temperature, axis=-1)
        
        soft_loss = F.kl_div(student_soft, teacher_soft, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Combined loss
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss
        
        return loss, hard_loss, soft_loss

# Usage
teacher = load_pretrained_model("PP-OCRv4_rec_large")
student = load_pretrained_model("PP-OCRv4_rec_mobile")
criterion = DistillationLoss(teacher, temperature=4.0, alpha=0.5)
```

---

## 8. **LoRA (Low-Rank Adaptation)** ⭐ For DeepSeek-OCR

### Why for VIN?
- DeepSeek-OCR is a large vision-language model
- Full fine-tuning requires too much memory
- LoRA trains only ~1% of parameters with similar accuracy

### Implementation

```python
import paddle
import paddle.nn as nn

class LoRALayer(nn.Layer):
    """
    Low-Rank Adaptation layer.
    
    Instead of updating W directly, we learn:
    W' = W + BA
    
    Where:
    - W: Original frozen weights [d_out, d_in]
    - B: Low-rank matrix [d_out, r]
    - A: Low-rank matrix [r, d_in]
    - r: Rank (typically 4-16)
    
    Trainable params: r × (d_in + d_out) << d_in × d_out
    """
    
    def __init__(self, original_layer, rank=8, alpha=16):
        super().__init__()
        
        self.original = original_layer
        # Freeze original weights
        for param in self.original.parameters():
            param.stop_gradient = True
        
        # Get dimensions
        if hasattr(original_layer, 'weight'):
            d_out, d_in = original_layer.weight.shape
        else:
            raise ValueError("Layer must have weight attribute")
        
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Low-rank matrices
        self.lora_A = self.create_parameter(
            shape=[rank, d_in],
            default_initializer=nn.initializer.KaimingUniform()
        )
        self.lora_B = self.create_parameter(
            shape=[d_out, rank],
            default_initializer=nn.initializer.Constant(0.0)
        )
    
    def forward(self, x):
        # Original forward
        original_out = self.original(x)
        
        # LoRA adaptation: x @ A^T @ B^T
        lora_out = x @ self.lora_A.T @ self.lora_B.T
        
        return original_out + self.scaling * lora_out


def apply_lora(model, rank=8, target_modules=['q_proj', 'v_proj']):
    """
    Apply LoRA to specific modules in the model.
    
    For vision-language models, typically target:
    - Attention: q_proj, k_proj, v_proj, o_proj
    - MLP: up_proj, down_proj
    """
    for name, module in model.named_sublayers():
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                parent = model.get_sublayer(parent_name)
                
                lora_layer = LoRALayer(module, rank=rank)
                setattr(parent, child_name, lora_layer)
                print(f"Applied LoRA to {name}")
    
    # Count trainable params
    trainable = sum(p.numel() for p in model.parameters() if not p.stop_gradient)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable with LoRA: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    
    return model

# Usage for DeepSeek-OCR
from ocr_providers import DeepSeekOCRProvider

provider = DeepSeekOCRProvider()
provider.initialize()  # Load model

# Apply LoRA
model = apply_lora(
    provider._model,
    rank=16,
    target_modules=['q_proj', 'v_proj', 'k_proj', 'o_proj']
)

# Train with very small dataset
# LoRA can work with as few as 100-500 examples!
```

---

## Recommended Training Configuration

### For Different Dataset Sizes

| Dataset Size | Recommended Technique | Epochs | Learning Rate |
|--------------|----------------------|--------|---------------|
| 100-500 images | LoRA (rank=8) | 10-20 | 1e-4 |
| 500-2000 images | Frozen Backbone | 30-50 | 1e-3 |
| 2000-5000 images | Gradual Unfreezing | 50-80 | 5e-4 |
| 5000-10000 images | Discriminative LR | 80-100 | 5e-4 |
| 10000+ images | Full Fine-Tuning | 100-200 | 1e-4 |

### Optimal Config for ~11K VIN Images

```yaml
# configs/vin_finetune_optimal.yml

Global:
  use_gpu: true
  epoch_num: 100
  use_amp: true
  amp_level: O1

Optimizer:
  name: AdamW  # Better than Adam for fine-tuning
  beta1: 0.9
  beta2: 0.999
  weight_decay: 0.01  # L2 regularization
  lr:
    name: CosineAnnealingWarmRestarts  # Better than simple cosine
    learning_rate: 0.0003
    warmup_epoch: 5
    T_0: 20  # Restart every 20 epochs
    T_mult: 2  # Double period after each restart

# Discriminative learning rates
FineTuning:
  technique: discriminative_lr
  backbone_lr_mult: 0.1
  neck_lr_mult: 0.5
  head_lr_mult: 1.0

# Data augmentation
Augmentation:
  prob: 0.5
  brightness_range: [0.7, 1.3]
  rotation_range: [-5, 5]
  blur_prob: 0.3
  noise_prob: 0.3
  perspective_prob: 0.3
  jpeg_quality_range: [50, 95]

# Training
Train:
  loader:
    batch_size_per_card: 64
    num_workers: 8
  dataset:
    transforms:
      - VINAugmentation:
          prob: 0.5
      - RecConAug:
          prob: 0.3
      - LabelSmoothing:
          smoothing: 0.1
```

---

## Summary: Which Technique to Use?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FINE-TUNING TECHNIQUE SELECTION                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  START                                                                      │
│    │                                                                        │
│    ▼                                                                        │
│  ┌─────────────────────────┐                                               │
│  │ Dataset Size?           │                                               │
│  └────────────┬────────────┘                                               │
│               │                                                             │
│    ┌──────────┼──────────┬──────────────┐                                  │
│    ▼          ▼          ▼              ▼                                  │
│  <500      500-2K      2K-10K        >10K                                  │
│    │          │          │              │                                  │
│    ▼          ▼          ▼              ▼                                  │
│  LoRA      Frozen     Gradual        Full                                  │
│            Backbone   Unfreezing   Fine-Tuning                             │
│    │          │          │              │                                  │
│    └──────────┴──────────┴──────────────┘                                  │
│                          │                                                  │
│                          ▼                                                  │
│              ALWAYS ADD:                                                    │
│              ✓ Mixed Precision (AMP)                                       │
│              ✓ Data Augmentation                                           │
│              ✓ Label Smoothing                                             │
│              ✓ Cosine LR Schedule                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Quick Commands

```bash
# 1. For small dataset (<500 images) - Use LoRA
python finetune_paddleocr.py --config configs/vin_lora_config.yml --technique lora

# 2. For medium dataset (500-5000 images) - Frozen backbone + gradual unfreeze
python finetune_paddleocr.py --config configs/vin_finetune_config.yml --technique gradual

# 3. For large dataset (>5000 images) - Full fine-tuning with discriminative LR
python finetune_paddleocr.py --config configs/vin_finetune_config.yml --technique full
```
