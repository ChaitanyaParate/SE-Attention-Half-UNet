# Attention Half U-Net with SE Blocks & Attention Gates  

# 1. Introduction

This repository implements a **fully custom, lightweight medical image segmentation model** called the **Attention Half U-Net (Att-H-UNet)**.  
It is inspired by Half U-Net, Attention U-Net, and SE-Net, and is optimized for:

- **Medical image segmentation (primarily polyp segmentation)**  
- **GPU efficiency and low latency**  
- **Better skip‑connection refinement through attention gates**  
- **Stronger channel recalibration via Squeeze-and-Excitation (SE) blocks**  

The encoder is deeper and more expressive, while the decoder is intentionally shallow — creating a **Half U-Net** that preserves strong performance with significantly fewer parameters.

---

# 2. Key Features

### ✅ Lightweight but powerful  
Encoder-heavy design with reduced decoder complexity.

### ✅ Attention Gates  
Suppress irrelevant spatial features before merging skip connections.

### ✅ Squeeze-and-Excitation (SE) Blocks  
Perform channel-wise feature recalibration.

### ✅ Custom BCE + Dice Loss  
Balances region accuracy and segmentation overlap.

### ✅ Cosine Annealing Warm Restarts  
Stabilizes training and allows fast convergence.

### ✅ Mixed Precision (AMP) Training  
Improved speed and reduced VRAM usage.

### ✅ Modular Codebase  
Each component (dataset, model, loss, utils) is neatly separated for easy modification.

---

# 3. Repository Structure

```
root/
│
├── dataset.py      → Custom PyTorch Dataset class, image & mask pipeline  
├── loss.py         → Dice Loss, BCE-Dice hybrid loss  
├── model.py        → Attention Half U-Net, SEBlock, AttentionGate  
├── train.py        → Training loop, validation, scheduler, AMP  
└── utils.py        → Checkpoints, loaders, metrics, utilities
```

Each file is self-contained and can be reused in other segmentation projects.

---

# 4. Architecture Overview

The **Attention Half U-Net** consists of:

- **Encoder:**  
  - Multiple DoubleConv blocks  
  - SE block after each encoder stage  
  - Increasing feature depth (e.g., 32 → 64 → 128 → 256)

- **Bottleneck:**  
  - High-level context extraction  
  - Transpose convolution to upsample

- **Decoder:**  
  - Minimal depth  
  - Attention gate on each skip connection  
  - Fusion of attention-filtered skip tensors

<p align="center">
  <img src="https://github.com/user-attachments/assets/04c7a5ac-42c2-419d-8122-1a40a0a45249" width="700">
</p>
---

# 5. Dataset Requirements

The model is designed for **binary medical segmentation** datasets like Kvasir-SEG.

### Folder structure
```
dataset/
 ├─ images/
 └─ masks/
```

### Images:  
- RGB  
- Any resolution (upsampled/cropped to 256×256 internally)

### Masks:  
- Single-channel  
- Values: **0 or 255**  
- `dataset.py` automatically binarizes them → {0,1}

---

# 6. Installation

Install dependencies:

```bash
pip install torch torchvision albumentations numpy tqdm pillow
```

---

# 7. Training

Run:

```bash
python train.py
```

### Important parameters in `train.py`:

| Parameter | Description |
|----------|-------------|
| `LEARNING_RATE` | Default 1e-4 |
| `BATCH_SIZE` | Default 12 |
| `IMAGE_HEIGHT`, `IMAGE_WIDTH` | Set crop/resize |
| `NUM_EPOCHS` | Training duration |
| `DEVICE` | "cuda" or "cpu" |
| `WEIGHT_DECAY` | L2 regularization |
| `PIN_MEMORY` | DataLoader optimization |

### Training features:
- AMP (automatic mixed precision)  
- AdamW with CosineAnnealingWarmRestarts  
- Auto‑checkpoint saving  
- Dice history logged

---

# 8. Evaluation

Validation metrics computed using `check_accuracy()`:

- **Dice Coefficient**
- **Pixel-wise Accuracy**

---

# 10. Results & Performance

<p align="center">
  <img src="https://github.com/user-attachments/assets/d38af22c-bfb0-410d-90ed-f5c2a77011f4" width="700">
</p>


Results on **Kvasir‑SEG**:

| Metric | Expected Range |
|--------|----------------|
| Dice Score | **0.90 – 0.94** |
| Pixel Accuracy | 95–97% |

The model aims to match larger U-Net architectures with **significantly fewer parameters**.

---

# 11. Design Choices & Rationale

### Why Half U-Net?
- Faster  
- Less memory  
- Sufficient for many medical segmentation tasks  

### Why Attention Gates?
Skip connections often carry irrelevant features.  
AG suppresses noisy activations → better mask boundaries.

### Why SE Blocks?
Helps the encoder focus on meaningful anatomical features.

### Why BCE + Dice Loss?
- BCE is stable for training  
- Dice provides spatial overlap sensitivity  
- Combined loss performs best in segmentation tasks

---

# 12. Future Improvements

- Add Swin Transformer encoder  
- Add ONNX Runtime export  
- Add Intersection over Union (IoU)

---

## 🧑‍💻 Author

**Chaitanya Parate**  
B.Tech in Computer Science and Engineering @ MIT-WPU, Pune  
Passionate about AI, ML, Deep Learning, and Computer Vision
