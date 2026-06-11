# Attention Half U-Net with SE Blocks & Attention Gates

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white"/>
  <img src="https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Task-Medical%20Image%20Segmentation-blue?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/Dataset-Kvasir--SEG-green?style=for-the-badge"/>
</p>

---

## Overview

This repository implements a fully custom, lightweight medical image segmentation model called the **Attention Half U-Net (Att-H-UNet)**. It is designed from scratch in PyTorch, drawing architectural inspiration from [Half U-Net](https://arxiv.org/abs/2205.07405), [Attention U-Net](https://arxiv.org/abs/1804.03999), and [Squeeze-and-Excitation Networks (SE-Net)](https://arxiv.org/abs/1709.01507).

The model is primarily evaluated on **polyp segmentation** using the [Kvasir-SEG](https://datasets.simula.no/kvasir-seg/) dataset. Its core design philosophy is **encoder-heavy, decoder-light** — achieving competitive segmentation performance with a significantly smaller parameter footprint than standard U-Net variants.

---

## Key Features

| Feature | Description |
|---|---|
| 🏗️ **Half U-Net Design** | Deep 4-stage encoder with a minimal single-stage decoder |
| 🎯 **Attention Gates (AG)** | Suppress irrelevant spatial activations on each skip connection |
| 🔌 **SE Blocks** | Channel-wise feature recalibration after every encoder stage |
| ⚖️ **BCE + Dice Loss** | Hybrid loss balancing pixel stability and region overlap |
| 🔄 **Cosine Annealing Warm Restarts** | Dynamic LR schedule for better convergence |
| ⚡ **AMP Training** | `bfloat16` mixed precision for reduced VRAM and faster training |
| 📐 **Gradient Clipping** | Norm clamped to 1.0 for stable training |
| 💾 **Best-Dice Checkpointing** | Saves only the checkpoint that exceeds the best validation Dice |

---

## Architecture

### Design Principle

The encoder is deep and expressive while the decoder is intentionally shallow. Rather than a symmetric decoder with multiple upsampling stages, all four skip connections are fused together at a single aggregation step — hence "Half" U-Net.

```
Input (3 × H × W)
      │
  ┌───▼────────────────────────────────────────┐
  │  Encoder Stage 1: DoubleConv → SE Block    │ → skip₁ (32ch)
  │         ↓ MaxPool                          │
  │  Encoder Stage 2: DoubleConv → SE Block    │ → skip₂ (64ch)
  │         ↓ MaxPool                          │
  │  Encoder Stage 3: DoubleConv → SE Block    │ → skip₃ (128ch)
  │         ↓ MaxPool                          │
  │  Encoder Stage 4: DoubleConv → SE Block    │ → skip₄ (256ch)
  │         ↓ MaxPool                          │
  │       Bottleneck (DoubleConv + Conv)        │
  └───────────────────────────────────────────-┘
      │ ConvTranspose2d (upsample)
      │
  ┌───▼──────────────────────────────────────────────────┐
  │  Decoder: Iterate over reversed skip connections     │
  │    For each skip_i:                                  │
  │      AttentionGate(g=decoder_feat, x=skip_i) → att_i │
  │      decoder_feat = decoder_feat + att_i             │
  │      (if not last) ConvTranspose2d → upsample        │
  └──────────────────────────────────────────────────────┘
      │
  final_conv (1×1) → Output mask (1 × H × W)
```

### Component Breakdown

**`SEBlock`** — Squeeze-and-Excitation  
Performs adaptive channel recalibration: global average pooling → bottleneck FC (reduction=8) → sigmoid gate → channel-wise scaling.

**`DoubleConv`** — Convolutional Block  
Two sequential `Conv2d(3×3) → BatchNorm2d → ReLU` layers. Used throughout the encoder and bottleneck.

**`AttentionGate`** — Spatial Attention  
Takes the gating signal `g` (from the decoder) and the skip connection `x`. Learns a spatial attention map `ψ` via 1×1 convolutions and sigmoid, then returns `x * ψ` — suppressing irrelevant spatial regions before fusion.

**`Att_H_UNET`** — Full Model  
- Encoder: 4 × `DoubleConv` + `SEBlock` + `MaxPool`
- Bottleneck: `DoubleConv` + pointwise `Conv2d`
- Single `ConvTranspose2d` upsample from bottleneck
- Decoder loop: 4 × `AttentionGate` + additive fusion + optional `ConvTranspose2d`
- Output: 1×1 `Conv2d` projecting to a single-channel mask logit

### Architecture Diagram

<p align="center">
  <img src="docs/Architecture.png" width="750" alt="Attention Half U-Net Architecture">
</p>

---

## Repository Structure

```
SE-Attention-Half-UNet/
│
├── model.py          # Att_H_UNET, SEBlock, AttentionGate, DoubleConv
├── train.py          # Training loop, AMP, scheduler, checkpointing
├── loss.py           # dice_loss(), BCEDiceLoss (BCE + Dice hybrid)
├── dataset.py        # TrainDataset — PyTorch Dataset for images & masks
├── utils.py          # save/load checkpoint, get_loaders, check_accuracy
├── plot.py           # Plots Dice score history from dice_history.pt
│
├── Kvasir-SEG/       # Training data (not tracked in git)
│   ├── images/
│   └── masks/
│
├── sessile-main-Kvasir-SEG/   # Validation data (not tracked in git)
│   ├── images/
│   └── masks/
│
├── docs/
│   ├── Architecture.png       # Model architecture diagram
│   └── Performance.png        # Dice score training curve
│
├── my_checkpoint.pth.tar      # Saved model weights (not tracked in git)
└── dice_history.pt            # Per-epoch Dice score log (not tracked in git)
```

---

## Dataset

The model is trained and validated on subsets of the **Kvasir-SEG** polyp segmentation benchmark.

| Split | Source |
|---|---|
| **Training** | `Kvasir-SEG` (1000 polyp images with pixel-level masks) |
| **Validation** | `sessile-main-Kvasir-SEG` (sessile polyp subset) |

**Dataset format:**
```
dataset/
├── images/    # RGB JPEG/PNG endoscopy frames
└── masks/     # Grayscale PNG masks (pixel values: 0 or 255)
```

**Preprocessing pipeline (`dataset.py`):**
- Images: loaded as RGB, passed through Albumentations transforms
- Masks: loaded as grayscale `float32`, divided by 255 then binarized at 0.5 → final values are `{0.0, 1.0}`
- All inputs resized to **256 × 256** before training

**Training augmentations (`train.py`):**
- Random rotation (±35°, always applied)
- Random horizontal flip (p=0.5)
- Random vertical flip (p=0.1)
- Pixel normalization to [0, 1]

---

## Installation

```bash
git clone https://github.com/ChaitanyaParate/SE-Attention-Half-UNet.git
cd SE-Attention-Half-UNet

pip install torch torchvision albumentations numpy tqdm pillow matplotlib
```

> Requires Python ≥ 3.8 and a CUDA-capable GPU for practical training times.

---

## Training

```bash
python train.py
```

### Hyperparameters

| Parameter | Value | Description |
|---|---|---|
| `LEARNING_RATE` | `1e-4` | Initial AdamW learning rate |
| `WEIGHT_DECAY` | `1e-4` | L2 regularization |
| `BATCH_SIZE` | `12` | Images per gradient step |
| `NUM_EPOCHS` | `120` | Training duration |
| `IMAGE_HEIGHT/WIDTH` | `256 × 256` | Input resolution |
| `DEVICE` | `cuda` | Target compute device |
| `PIN_MEMORY` | `True` | Faster CPU→GPU transfer |
| `NUM_WORKERS` | `2` | DataLoader worker threads |

### Optimizer & Scheduler

- **Optimizer:** `AdamW` with `weight_decay=1e-4`
- **Scheduler:** `CosineAnnealingWarmRestarts`
  - `T_0=5` (restart every 5 epochs)
  - `T_mult=2` (restart period doubles each cycle)
  - `eta_min=1e-7` (minimum LR floor)

### Training Loop Details

- Mixed precision via `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`
- Gradient scaler via `torch.amp.GradScaler('cuda')`
- Gradient norm clipped to `1.0` before each optimizer step
- Checkpoint saved only when validation Dice exceeds the current best (`best_dice=0.9` threshold)
- Full Dice history saved to `dice_history.pt` after each epoch

---

## Evaluation

Validation metrics are computed by `check_accuracy()` in `utils.py` after each epoch:

```
Metric          Formula
─────────────── ──────────────────────────────────────────────────────────────
Dice Score      2 × |P ∩ G| / (|P| + |G| + ε)   (ε = 1e-8)
Pixel Accuracy  (# correct pixels) / (# total pixels)
```

- Predictions are thresholded at **0.5** (after sigmoid) to obtain binary masks
- Model is set to `eval()` mode during validation, then restored to `train()` mode

### Plotting Training Progress

```bash
python plot.py
```

Loads `dice_history.pt` and renders a Dice score vs. Epoch curve.

---

## Results

### Training Curve

<p align="center">
  <img src="docs/Performance.png" width="750" alt="Dice Score vs Epoch on Kvasir-SEG">
</p>

The model shows stable improvement across 150 epochs, converging to a Dice score of **~0.93** on the validation set, consistent with strong polyp segmentation performance.

### Benchmark Performance on Kvasir-SEG

| Metric | Score |
|---|---|
| **Dice Score** | **~0.93** |
| **Pixel Accuracy** | **~96–97%** |

---

## Loss Function

The `BCEDiceLoss` combines two complementary objectives with equal weighting:

```
L = 0.5 × BCE + 0.5 × Dice
```

- **BCE (`BCEWithLogitsLoss`):** Pixel-level cross-entropy — numerically stable due to fused sigmoid. Ensures per-pixel class balance.
- **Dice Loss (`dice_loss`):** Region-level overlap — directly optimizes the Dice coefficient, making the model sensitive to foreground shape.

The combined loss avoids the individual failure modes of each: BCE alone ignores mask geometry; Dice alone can be unstable with very small masks.

---

## Design Rationale

### Why Half U-Net?
A full symmetric decoder adds significant parameter and compute cost with diminishing returns for many binary segmentation tasks. The half decoder fuses all skip connections at one level, reducing decoding cost while retaining fine-grained spatial information from all encoder stages via attention gates.

### Why Attention Gates?
Standard skip connections concatenate encoder and decoder features indiscriminately, including background noise. Attention gates learn a spatial saliency map conditioned on the current decoder state, selectively highlighting foreground-relevant spatial regions before fusion.

### Why SE Blocks in the Encoder?
After each DoubleConv block, the SE block performs global channel-wise recalibration. This allows the encoder to amplify anatomically meaningful feature channels (e.g., polyp texture or boundary contrast) and suppress uninformative ones — improving the quality of representations passed through skip connections.

### Why AdamW + Cosine Annealing Warm Restarts?
AdamW decouples weight decay from the gradient update, preventing L2 regularization from interfering with adaptive learning rates. Cosine Annealing Warm Restarts periodically resets the learning rate, helping the optimizer escape local minima and explore better loss landscape regions.

---

## Future Improvements

- [ ] Add IoU (Jaccard Index) as an additional evaluation metric  
- [ ] Export model to ONNX for cross-platform inference  
- [ ] Support multi-class segmentation (softmax output head)  
- [ ] Experiment with a Swin Transformer or ConvNeXt encoder  
- [ ] Add inference script with visualization for arbitrary input images  
- [ ] Benchmark against standard U-Net, Attention U-Net on the same splits

---

## License

This project is released under the [MIT License](LICENSE).

---

## Author

**Chaitanya Parate**  
B.Tech in Computer Science and Engineering @ MIT-WPU, Pune  
Passionate about AI, Deep Learning, and Computer Vision

[![GitHub](https://img.shields.io/badge/GitHub-ChaitanyaParate-181717?style=flat&logo=github)](https://github.com/ChaitanyaParate)
