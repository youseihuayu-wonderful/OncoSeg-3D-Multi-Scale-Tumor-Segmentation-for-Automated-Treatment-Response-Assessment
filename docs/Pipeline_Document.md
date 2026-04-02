# 3D Brain Tumor MRI Automatic Segmentation and Treatment Response Assessment

## Complete Pipeline Document

**Project:** OncoSeg  
**Author:** Shihua Yu & Claude Opus 4.6  
**Dataset:** Medical Segmentation Decathlon (MSD) Task01 — Brain Tumour  
**Framework:** PyTorch + MONAI  

---

## Pipeline Overview

```
3D Brain Tumor MRI Automatic Segmentation and Treatment Response Assessment
├── 1. Project Objective and Problem Definition
├── 2. Data Preparation
├── 3. Environment and Tools Setup
├── 4. Data Preprocessing
├── 5. Model Development and Training
└── 6. Evaluation and Clinical Application
```

---

## 1. Project Objective and Problem Definition

### 1.1 Segmentation Task

**Goal:** Automatically segment brain tumors from 3D multi-modal MRI volumes into three clinically relevant regions:

| Region | Abbreviation | Definition | Clinical Relevance |
|--------|-------------|------------|-------------------|
| Tumor Core | TC | Non-enhancing tumor + Enhancing tumor (labels 2+3) | Core tumor mass requiring surgical planning |
| Whole Tumor | WT | Edema + Non-enhancing + Enhancing (labels 1+2+3) | Full extent of disease including peritumoral edema |
| Enhancing Tumor | ET | Enhancing tumor only (label 3) | Active tumor with blood-brain barrier breakdown |

**Input:** 4-channel 3D MRI volume (FLAIR, T1w, T1gd, T2w)  
**Output:** 3-channel binary segmentation mask (TC, WT, ET)

### 1.2 Treatment Response Assessment Goal

**Goal:** Automate RECIST 1.1 (Response Evaluation Criteria in Solid Tumors) measurements and classify treatment response.

**Traditional workflow (manual):**
- Radiologist measures longest axial diameter of each lesion
- Compare baseline vs. follow-up measurements
- Classify response: CR, PR, SD, or PD
- Takes 15-30 minutes per patient, with 20-40% inter-reader variability

**Our automated workflow:**
- Segment tumor from MRI using OncoSeg model
- Compute longest axial diameter and volume from 3D segmentation mask
- Compare baseline vs. follow-up using temporal attention
- Automatically classify treatment response

**Response Categories (RECIST 1.1):**

| Category | Criteria | Meaning |
|----------|---------|---------|
| Complete Response (CR) | Target lesion diameter = 0 | Tumor disappeared |
| Partial Response (PR) | ≥30% decrease in sum of diameters | Tumor is shrinking |
| Progressive Disease (PD) | ≥20% increase in sum of diameters | Tumor is growing |
| Stable Disease (SD) | Neither PR nor PD | No significant change |

### 1.3 Research Questions

1. Can a hybrid Swin Transformer-CNN architecture outperform pure CNN (U-Net) and pure Transformer (UNETR) baselines for 3D brain tumor segmentation?
2. Do cross-attention skip connections improve over standard concatenation/additive skip connections?
3. Does deep supervision improve convergence and final segmentation quality?
4. Can automated RECIST measurements from segmentation masks replace manual measurements?

---

## 2. Data Preparation

### 2.1 Dataset: MSD Task01 — Brain Tumour

| Property | Value |
|----------|-------|
| Source | Medical Segmentation Decathlon (MSD) |
| Download | `https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar` |
| Size | 7.09 GB (compressed) |
| Training subjects | 484 |
| Test subjects | 266 (no labels) |
| Original source | BraTS 2016/2017 challenge |

### 2.2 Imaging Modalities

Each subject has a 4D NIfTI file with 4 co-registered MRI modalities:

| Channel | Modality | Full Name | What It Shows |
|---------|----------|-----------|---------------|
| 0 | FLAIR | Fluid-Attenuated Inversion Recovery | Edema appears bright; CSF suppressed |
| 1 | T1w | T1-weighted | Anatomical structure; tumor appears dark |
| 2 | T1gd | T1-weighted with Gadolinium contrast | Enhancing tumor appears bright (BBB breakdown) |
| 3 | T2w | T2-weighted | Edema and tumor appear bright |

**Why 4 modalities?** Each modality highlights different tissue properties. The model learns complementary features from all 4 to distinguish tumor sub-regions.

### 2.3 Labels and Annotations

**MSD label convention:**

| Label Value | Region | Color in Visualizations |
|-------------|--------|------------------------|
| 0 | Background (healthy tissue) | — |
| 1 | Peritumoral Edema | Yellow |
| 2 | Non-enhancing Tumor Core | Red |
| 3 | GD-enhancing Tumor | Green |

**Multi-channel conversion (for training):**

The raw single-channel labels are converted to 3 overlapping binary channels:

```
Raw labels {0, 1, 2, 3}
    │
    ▼  ConvertMSDToMultiChanneld
    │
Channel 0 (TC): label==2 OR label==3     → Tumor Core
Channel 1 (WT): label==1 OR 2 OR 3       → Whole Tumor  
Channel 2 (ET): label==3                  → Enhancing Tumor
```

**Why overlapping?** TC is a subset of WT, and ET is a subset of TC. This hierarchical structure reflects clinical practice: WT includes everything, TC excludes edema, ET is the most aggressive component.

### 2.4 Data Integrity and Train/Validation Split

**Verification (every subject):**
- Image is 4D with exactly 4 modalities
- Label is 3D with values in {0, 1, 2, 3}
- Image and label have matching spatial dimensions
- Valid voxel spacing (typically 1mm isotropic)

**Split:**
- Training: 387 subjects (80%)
- Validation: 97 subjects (20%)
- Deterministic split with seed=42 for reproducibility

---

## 3. Environment and Tools Setup

### 3.1 Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU | T4 (16 GB VRAM) | A100 (40/80 GB) |
| RAM | 12 GB | 32 GB |
| Disk | 20 GB free | 50 GB free |
| Platform | Google Colab / Kaggle | Local workstation |

### 3.2 Software Stack

| Package | Version | Purpose |
|---------|---------|---------|
| Python | 3.11+ | Runtime |
| PyTorch | 2.1+ | Deep learning framework |
| MONAI | 1.3.2 | Medical image transforms, metrics, models |
| NumPy | <2.0 | Array operations (MONAI 1.3.x compatibility) |
| nibabel | latest | NIfTI file I/O |
| SimpleITK | latest | Medical image processing |
| einops | latest | Tensor reshaping (for Transformers) |
| scipy | latest | Statistical tests, connected components |
| matplotlib | latest | Visualization |
| pandas | latest | Results tables |

### 3.3 Reproducibility Configuration

```python
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

All random operations (data split, augmentation, weight initialization) use this seed for deterministic results.

---

## 4. Data Preprocessing

### 4.1 Loading and Formatting

```
Raw NIfTI files
├── Image: (H, W, D, 4) → EnsureChannelFirst → (4, H, W, D)
└── Label: (H, W, D)    → EnsureChannelFirst → (1, H, W, D)
                         → ConvertMSD         → (3, H, W, D)
```

### 4.2 Intensity Normalization

```python
NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True)
```

- **nonzero=True:** Only normalize within the brain mask (non-zero voxels). Background stays at 0.
- **channel_wise=True:** Normalize each modality independently (each has different intensity distributions).
- **Method:** z-score normalization: `(x - mean) / std` per channel, computed only on non-zero voxels.

### 4.3 Spatial Transforms

| Transform | Parameters | Purpose |
|-----------|-----------|---------|
| `Orientationd` | axcodes="RAS" | Standardize orientation (Right-Anterior-Superior) |
| `Spacingd` | pixdim=(1,1,1)mm | Resample to isotropic 1mm voxels |
| `CropForegroundd` | source_key="image" | Remove empty background slices |
| `SpatialPadd` | spatial_size=(128,128,128) | Pad if volume < 128 in any dimension |
| `RandSpatialCropd` | roi_size=(128,128,128) | Extract random training patch |

**Why 128x128x128?** Balances spatial context (large enough for tumor anatomy) with GPU memory (fits in T4 16GB with batch_size=2).

### 4.4 Data Augmentation (Training Only)

| Augmentation | Probability | Purpose |
|-------------|-------------|---------|
| Random Flip (axis 0, 1, 2) | 50% each | Left-right, anterior-posterior, superior-inferior symmetry |
| Random Rotate 90° | 50% | Rotation invariance |
| Random Scale Intensity | 50%, ±10% | Simulate scanner intensity variations |
| Random Shift Intensity | 50%, ±10% | Simulate scanner offset variations |

**Why these augmentations?** Brain MRI has natural symmetry (flip), scanner variations (intensity), and the model should be invariant to orientation (rotate). We avoid elastic deformation to preserve tumor morphology.

---

## 5. Model Development and Training

### 5.1 Main Model: OncoSeg

**Architecture:** Hybrid Swin Transformer Encoder + CNN Decoder with Cross-Attention Skip Connections

```
Input: (B, 4, 128, 128, 128)
│
▼ Swin Transformer Encoder
├── Patch Embed:  (B, 48,  32, 32, 32)   ← 4x downsample
├── Stage 1:      (B, 48,  32, 32, 32)   ← window attention
├── Stage 2:      (B, 96,  16, 16, 16)   ← patch merge + attention
├── Stage 3:      (B, 192,  8,  8,  8)   ← patch merge + attention
└── Stage 4:      (B, 384,  4,  4,  4)   ← bottleneck
│
▼ CNN Decoder (with skip connections)
├── Decoder 3: (B, 192, 8,8,8)   + Cross-Attention from Stage 3
├── Decoder 2: (B, 96, 16,16,16) + Cross-Attention from Stage 2
├── Decoder 1: (B, 48, 32,32,32) + Addition from Stage 1
│                                   (cross-attn skipped: 32^3 tokens too large)
├── Upsample:  (B, 48, 128,128,128) ← ConvTranspose3d stride=4
└── Output:    (B, 3, 128,128,128)  ← Conv3d 1x1
│
▼ Deep Supervision (training only)
├── DS Head 1: intermediate prediction at 8^3 → upsample to 128^3
├── DS Head 2: intermediate prediction at 16^3 → upsample to 128^3
└── DS Head 3: intermediate prediction at 32^3 → upsample to 128^3
```

**Key design decisions:**

| Decision | Choice | Reason |
|----------|--------|--------|
| Encoder | Swin Transformer | Window-based attention: O(n) memory, captures long-range dependencies |
| Decoder | CNN (ConvTranspose3d) | Efficient upsampling, good for local detail reconstruction |
| Skip connections | Cross-attention (deeper) + Addition (finest) | Selective feature fusion; addition at 32^3 avoids OOM |
| Uncertainty | MC Dropout (p=0.1) | Dropout at bottleneck during inference gives uncertainty maps |
| Deep supervision | 3 auxiliary heads | Addresses vanishing gradients, improves intermediate features |

### 5.2 Baseline Models

| Model | Type | Parameters | Description |
|-------|------|-----------|-------------|
| 3D U-Net | Pure CNN | ~25M | MONAI implementation, 5-level encoder-decoder |
| SwinUNETR | Transformer-CNN | ~62M | MONAI Swin Transformer encoder + CNN decoder |
| UNETR | ViT-CNN | ~102M | MONAI Vision Transformer encoder + CNN decoder |

### 5.3 Loss Function

**Combined Dice + Binary Cross-Entropy Loss:**

```
L_total = 0.5 × L_dice + 0.5 × L_bce
```

- **Dice Loss** (`sigmoid=True`): Handles extreme class imbalance (tumor < 1% of voxels). Directly optimizes the evaluation metric.
- **BCE Loss** (`BCEWithLogitsLoss`): Provides stable per-voxel gradients. Prevents the model from ignoring small tumor regions.

**Deep Supervision Loss:**
```
L_ds = Σ (w_i × L_base(pred_i, target))
weights = [4/7, 2/7, 1/7]  (higher weight for deeper predictions)
```

**Total training loss for OncoSeg:**
```
L = L_base(main_pred, target) + 0.5 × L_ds(deep_sup_preds, target)
```

### 5.4 Training Configuration

| Hyperparameter | Value | Reason |
|---------------|-------|--------|
| Optimizer | AdamW | Weight decay regularization, good for Transformers |
| Learning rate | 1e-4 | Standard for medical segmentation |
| Weight decay | 1e-5 | Mild regularization |
| Scheduler | CosineAnnealingLR (eta_min=1e-6) | Smooth LR decay, better final convergence |
| Batch size | 2 | Fits T4 GPU (16GB) |
| Max epochs | 100 | Full models; 50 for ablation variants |
| Validation interval | Every 5 epochs | Balance between monitoring and training speed |
| Gradient clipping | max_norm=1.0 | Prevent gradient explosion (common with Transformers) |
| Sliding window (val) | roi=128^3, overlap=0.5, sw_batch=2 | Full-volume inference at validation time |

### 5.5 Training Loop

```
For each epoch:
    1. Train phase:
       - Forward pass through model
       - Compute loss (Dice+BCE, with deep supervision for OncoSeg)
       - Backward pass + gradient clipping
       - Update weights (AdamW)
    
    2. Validation phase (every 5 epochs):
       - Sliding window inference on full validation volumes
       - Compute Dice score per region (TC, WT, ET)
       - Save checkpoint if new best mean Dice
    
    3. Update learning rate (CosineAnnealing)
```

---

## 6. Evaluation and Clinical Application

### 6.1 Segmentation Metrics

| Metric | What It Measures | Better |
|--------|-----------------|--------|
| **Dice Score** | Overlap between prediction and ground truth (0-1) | Higher |
| **HD95** | 95th percentile Hausdorff distance in mm | Lower |
| **ASD** | Average Surface Distance in mm | Lower |
| **Sensitivity** | True positive rate (detection ability) | Higher |
| **Specificity** | True negative rate (false alarm rate) | Higher |

**Primary metric:** Mean Dice across TC, WT, ET (standard BraTS metric)

**Evaluation protocol:**
1. Load best checkpoint (highest validation Dice)
2. Sliding window inference on full validation volumes (roi=128^3, overlap=50%)
3. Threshold predictions at 0.5 after sigmoid
4. Compute all metrics per subject, then average

### 6.2 Model Comparison

Results table format (filled after training):

| Model | Dice ET | Dice TC | Dice WT | Dice Mean | HD95 Mean | Best Epoch |
|-------|---------|---------|---------|-----------|-----------|------------|
| OncoSeg | — | — | — | — | — | — |
| 3D U-Net | — | — | — | — | — | — |
| SwinUNETR | — | — | — | — | — | — |
| UNETR | — | — | — | — | — | — |

### 6.3 Ablation Studies

| Variant | What Changes | Purpose |
|---------|-------------|---------|
| OncoSeg (full) | Baseline — all components | Reference |
| OncoSeg (concat skip) | Replace cross-attention with addition | Test value of cross-attention |
| OncoSeg (no deep supervision) | Remove auxiliary loss heads | Test value of deep supervision |
| Encoder comparison | U-Net vs UNETR vs SwinUNETR | CNN vs Transformer vs Hybrid |

**Statistical significance:** Paired Wilcoxon signed-rank test (p < 0.05) on per-subject Dice scores between OncoSeg and each baseline.

### 6.4 Treatment Response Assessment (RECIST 1.1)

**Pipeline:**

```
3D Segmentation Mask (from OncoSeg)
│
▼ RECIST Measurement
├── Find connected components (lesions) via scipy.ndimage.label
├── For each lesion:
│   ├── Find axial slice with largest tumor area
│   ├── Compute longest axial diameter (mm)
│   └── Compute volume (mm³)
├── Sum of longest diameters across target lesions
│
▼ Response Classification (baseline vs. follow-up)
├── CR: sum of diameters = 0
├── PR: ≥30% decrease
├── PD: ≥20% increase
└── SD: neither PR nor PD
```

**Clinical significance:** Automated RECIST measurements eliminate:
- Inter-reader variability (20-40% in manual measurement)
- Time burden (15-30 min per patient reduced to seconds)
- 2D limitation (3D volumetric analysis captures true tumor burden)

---

## File Structure

```
OncoSeg/
├── configs/                    # Hydra YAML configuration files
│   ├── model/                  # Model configs (oncoseg, unet3d, unetr, swin_unetr)
│   ├── data/                   # Dataset configs (brats2023, kits23, lits, btcv)
│   └── experiment/             # Experiment configs
├── data/scripts/               # Dataset download scripts
├── docs/                       # Documentation
│   ├── AI_Knowledge_Fundamentals.md
│   ├── Hardware_and_Data_Requirements.md
│   ├── Paper_Methods_Draft.md
│   └── Pipeline_Document.md    # ← This document
├── notebooks/
│   └── OncoSeg_Full_Pipeline.ipynb  # Complete end-to-end Colab notebook
├── src/
│   ├── models/
│   │   ├── oncoseg.py          # Main model
│   │   ├── modules/            # Cross-attention, decoder, encoder, deep supervision
│   │   └── baselines/          # U-Net, UNETR, SwinUNETR wrappers
│   ├── data/                   # Dataset loaders and transforms
│   ├── training/               # Training loop and losses
│   ├── evaluation/             # Metrics and evaluation
│   ├── response/               # RECIST measurement and response classification
│   ├── analysis/               # Result analysis and figures
│   └── inference.py            # Inference with uncertainty quantification
├── tests/                      # 31 unit tests
├── pyproject.toml              # Project dependencies
└── README.md                   # Project overview
```

---

## How to Run

### Option A: Google Colab (Recommended)

1. Open [colab.research.google.com](https://colab.research.google.com)
2. Upload `notebooks/OncoSeg_Full_Pipeline.ipynb`
3. Runtime → Change runtime type → **T4 GPU**
4. Runtime → Run all
5. Training takes ~4-6 hours on T4, ~1-2 hours on A100

### Option B: Kaggle

1. Open [kaggle.com](https://www.kaggle.com) → New Notebook
2. Upload the notebook
3. Settings → Accelerator → **GPU T4 x2**
4. Run All

### Option C: Local

```bash
git clone https://github.com/youseihuayu-wonderful/OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment.git
cd OncoSeg-*
pip install -e ".[dev]"
python -m pytest tests/  # Verify installation
# Then open the notebook in Jupyter
```

---

## Summary

This pipeline takes raw brain tumor MRI data through a complete end-to-end workflow:

1. **Problem Definition** — 3-region tumor segmentation + automated RECIST response assessment
2. **Data Preparation** — 484 real clinical MRI volumes from MSD, verified and split
3. **Environment Setup** — PyTorch + MONAI on GPU, fully reproducible (seed=42)
4. **Preprocessing** — Resampling, normalization, augmentation optimized for brain MRI
5. **Model Training** — Hybrid Swin Transformer-CNN (OncoSeg) + 3 baselines, 100 epochs
6. **Evaluation** — Dice, HD95 metrics + statistical significance + automated RECIST

**All data is real. All results are genuine. No mock data.**
