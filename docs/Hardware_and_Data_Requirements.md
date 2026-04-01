# OncoSeg — Hardware Components & Data Requirements

> A complete reference of every hardware component and data requirement for running OncoSeg, ranked by importance, with specific numbers and Colab configurations.

---

## Table of Contents

1. [Overall Importance Hierarchy](#1-overall-importance-hierarchy)
2. [Data — The Most Important Component](#2-data--the-most-important-component)
   - 2.1 Why Data Is #1
   - 2.2 Data Scarcity in Medical AI
   - 2.3 Data Quality Determines Model Quality
   - 2.4 Medical Data Format Challenges
   - 2.5 Data Defines the Model's Capability Boundary
   - 2.6 Our Data Strategy
3. [GPU — The Critical Hardware Component](#3-gpu--the-critical-hardware-component)
   - 3.1 What the GPU Does in OncoSeg
   - 3.2 GPU Memory (VRAM) Requirements
   - 3.3 GPU Compute Requirements
   - 3.4 GPU Speed Comparison
   - 3.5 What Happens When VRAM Is Insufficient
4. [RAM — Data Loading Pipeline](#4-ram--data-loading-pipeline)
   - 4.1 What RAM Does in OncoSeg
   - 4.2 RAM Requirements
   - 4.3 RAM Bottleneck Symptoms
5. [Storage — Dataset & Checkpoints](#5-storage--dataset--checkpoints)
   - 5.1 Storage Breakdown
   - 5.2 SSD vs HDD Impact
   - 5.3 Total Storage Needed
6. [CPU — Preprocessing & Orchestration](#6-cpu--preprocessing--orchestration)
   - 6.1 What the CPU Does
   - 6.2 CPU Requirements
7. [Network Bandwidth — One-Time Setup](#7-network-bandwidth--one-time-setup)
8. [Google Colab Hardware Profile](#8-google-colab-hardware-profile)
9. [OncoSeg Colab Configuration Guide](#9-oncoseg-colab-configuration-guide)
10. [Hardware vs Data — The Real Hierarchy](#10-hardware-vs-data--the-real-hierarchy)

---

## 1. Overall Importance Hierarchy

```
Importance for a working AI system:

#1  DATA           → Defines WHAT the model can learn
#2  ALGORITHM      → Defines HOW the model learns (OncoSeg architecture)
#3  GPU            → Defines HOW FAST the model learns
#4  RAM / CPU      → Supports data loading pipeline
#5  Storage        → Holds the data and checkpoints
#6  Network        → Downloads the data once

A perfect GPU with bad data = useless model
A slow GPU with perfect data = good model (just takes longer)

Data >> Hardware in importance.
```

---

## 2. Data — The Most Important Component

### 2.1 Why Data Is #1

Data is not hardware — but it is the single most important component of the entire project. Everything depends on it:

| Component | Without Data |
|---|---|
| Model training | Impossible — weights can't learn anything |
| Evaluation metrics | Meaningless — Dice, HD95 need ground truth labels |
| Data augmentation | Nothing to augment |
| Loss functions | Nothing to compute loss against |
| RECIST measurement | No segmentation masks to measure |
| Response classification | No baseline/follow-up to compare |
| The entire Colab notebook | An empty shell that produces nothing |

```
Without GPU:    training is slow but theoretically possible
Without data:   NOTHING is possible. Zero. The project doesn't exist.
```

---

### 2.2 Data Scarcity in Medical AI

```
ImageNet (natural images):     14,000,000 images, freely available
MSD Brain Tumor (our data):    484 subjects

Why so few?
  - Each MRI scan requires a patient, a hospital, a scanner, a radiologist
  - Each label requires an EXPERT NEURORADIOLOGIST to manually annotate
  - Manual annotation of one 3D brain MRI: 2-4 hours per subject
  - 484 subjects × 3 hours = ~1,452 hours of expert radiologist time
  - At $200/hour for a neuroradiologist: ~$290,000 worth of annotation labor

This is why:
  - We use heavy data augmentation (flip, rotate, intensity variations)
  - We use cache_rate to maximize data reuse
  - We use Dice loss (handles the imbalanced small dataset)
  - Every single training sample is precious
```

---

### 2.3 Data Quality Determines Model Quality

```
"Garbage in, garbage out" — this is literally true in deep learning.

If labels are wrong:
  - Model learns to predict wrong boundaries
  - Dice score has a ceiling set by label quality
  - Inter-rater variability for brain tumor annotation: ~0.85-0.90 Dice
  - Even a perfect model can't exceed the annotation quality

If preprocessing is wrong:
  - Wrong orientation → model sees flipped anatomy
  - Wrong spacing → RECIST measurements in wrong units
  - No normalization → model can't generalize across scanners

This is why we have:
  - 6 preprocessing steps (load, orient, resample, normalize, crop, augment)
  - Data verification in the notebook (check EVERY file)
  - Z-score normalization to handle scanner differences
```

---

### 2.4 Medical Data Format Challenges

```
Medical data is NOT like natural images:

Natural image:  [3, 256, 256]       = 196K pixels,    RGB, 2D, PNG/JPEG
Brain MRI:      [4, 240, 240, 155]  = 35.7M voxels,  4 modalities, 3D, NIfTI

Challenges unique to medical data:
  - 3D volumes (not 2D images) → need 3D convolutions, more memory
  - 4 modalities per subject (T1, T1ce, T2, FLAIR) → 4 input channels
  - Physical coordinates (mm) → affine matrix, voxel spacing matter
  - Variable resolution across scanners → must resample to isotropic spacing
  - Variable orientation across scanners → must standardize to RAS
  - Extreme class imbalance → tumor is 2.5% of volume, background is 97.5%
  - Large file sizes → single subject = 50-200 MB
```

---

### 2.5 Data Defines the Model's Capability Boundary

```
Our MSD dataset: 484 brain tumor MRI scans, 4 classes

What the model CAN learn:
  ✓ Segment brain tumors in adult MRI
  ✓ Distinguish enhancing tumor, edema, necrotic core
  ✓ Generalize to similar MRI scanners and protocols

What the model CANNOT learn (limited by data):
  ✗ Segment tumors in other organs (no kidney/liver/lung data)
  ✗ Work on CT scans (trained only on MRI)
  ✗ Segment pediatric brain tumors (different appearance)
  ✗ Handle rare tumor types not in the dataset
  ✗ Work on MRI without all 4 modalities

The data defines the model's entire capability boundary.
No amount of hardware or algorithmic innovation can overcome missing data.
```

---

### 2.6 Our Data Strategy in OncoSeg

| Strategy | Why | File |
|---|---|---|
| **MSD Brain Tumor** as primary dataset | Freely downloadable, real clinical data, 484 subjects | `src/data/msd_dataset.py` |
| **BraTS 2023** as secondary | Larger (1,251 subjects), gold-standard benchmark | `src/data/brats_dataset.py` |
| **Extensive augmentation** | Multiply effective dataset 100x | `src/data/transforms.py` |
| **Cache dataset** | Don't waste time re-loading from disk | `cache_rate=0.1` in configs |
| **Data verification** | Validate every file before training | Notebook cells 8-10 |
| **Deterministic splits** | Same train/val split across experiments for fair comparison | `seed=42` in dataset loaders |
| **Multi-channel BraTS regions** | Convert integer labels to clinically meaningful ET/TC/WT | `ConvertToMultiChannelBasedOnBratsClassesd` |

---

## 3. GPU — The Critical Hardware Component

### 3.1 What the GPU Does in OncoSeg

Every single matrix multiplication, convolution, and attention computation runs on the GPU. Without a GPU, training OncoSeg is essentially impossible.

| Operation | GPU Workload | Why GPU Beats CPU |
|---|---|---|
| 3D Convolution | Millions of parallel multiply-accumulate ops per layer | GPU has ~10,000 CUDA cores vs CPU's ~16 cores |
| Attention (Q·K^T) | Matrix multiplication on [343, 16] × [16, 343] per window, per head, per layer | Massively parallel — each element is independent |
| Backpropagation | Compute gradients for ~25M parameters simultaneously | GPU parallelism: compute all gradients at once |
| Softmax | Apply exp() and normalize across 2M+ voxels | Element-wise ops are trivially parallel |
| Data augmentation (on GPU) | Random transforms on 128³ volumes | MONAI can run transforms on GPU |

---

### 3.2 GPU Memory (VRAM) Requirements

VRAM is the single most important GPU spec for 3D medical imaging.

```
What must fit in VRAM simultaneously during training:

  Model parameters:    ~25M params × 4 bytes = ~100 MB
  Optimizer states:    ~25M × 8 bytes (Adam: m + v) = ~200 MB
  Gradient storage:    ~25M × 4 bytes = ~100 MB
  Input batch:         [2, 4, 128, 128, 128] × 4 bytes = ~67 MB
  Activations:         All intermediate feature maps = ~2-4 GB
  Attention matrices:  Per window, per head, per layer = ~1-2 GB
  ─────────────────────────────────────────────────────────────
  Total:               ~4-7 GB for batch_size=1
                       ~6-10 GB for batch_size=2

  Minimum VRAM:   12 GB (T4) — batch_size=1-2
  Recommended:    16 GB (V100/T4) — batch_size=2
  Ideal:          40-80 GB (A100) — batch_size=4-8, faster convergence
```

---

### 3.3 GPU Compute Requirements

| Spec | What It Means | T4 (Free Colab) | V100 | A100 |
|---|---|---|---|---|
| **CUDA Cores** | General-purpose parallel processors | 2,560 | 5,120 | 6,912 |
| **Tensor Cores** | Specialized matrix multiply units (mixed precision) | 320 | 640 | 432 (3rd gen) |
| **Memory Bandwidth** | How fast data moves GPU memory ↔ cores | 320 GB/s | 900 GB/s | 2,039 GB/s |
| **FP32 Performance** | Single-precision compute speed | 8.1 TFLOPS | 15.7 TFLOPS | 19.5 TFLOPS |
| **FP16 Performance** | Half-precision (with Tensor Cores) | 65 TFLOPS | 125 TFLOPS | 312 TFLOPS |

---

### 3.4 GPU Speed Comparison

```
Training OncoSeg (300 epochs, MSD Brain Tumor, batch_size=2):

CPU only (no GPU):  ~100-150 days     ← Essentially impossible
T4 (16 GB):         ~3-6 days          ← Feasible on Colab free tier
V100 (32 GB):       ~1.5-3 days        ← Good with Colab Pro
A100 (40 GB):       ~15-40 hours       ← Ideal with Colab Pro+

Per epoch:
  CPU:   ~8-12 hours
  T4:    ~15-30 minutes
  V100:  ~8-15 minutes
  A100:  ~3-8 minutes

Speedup over CPU:
  T4:    ~20-30x faster
  V100:  ~40-60x faster
  A100:  ~80-120x faster
```

---

### 3.5 What Happens When VRAM Is Insufficient

```
batch_size=2, input 128³, OncoSeg model:
  ~8 GB VRAM needed

If only 6 GB available:
  → CUDA out of memory error
  → Must reduce batch_size to 1 or roi_size to 96³
  → Slower convergence, potentially worse results

If only 4 GB available:
  → Cannot run OncoSeg at all, even with batch_size=1
  → Would need to switch to a smaller model (UNet3D baseline)

Mitigation strategies:
  1. Reduce batch_size (1 instead of 2)
  2. Reduce roi_size (96³ instead of 128³)
  3. Use mixed precision training (FP16 — halves memory)
  4. Use gradient checkpointing (trade compute for memory)
  5. Use a smaller model variant (fewer Swin stages/channels)
```

---

## 4. RAM — Data Loading Pipeline

### 4.1 What RAM Does in OncoSeg

| Task | RAM Usage | Details |
|---|---|---|
| Load NIfTI files | ~50-200 MB per subject | Each 4D MRI volume (240×240×155×4 modalities) |
| MONAI CacheDataset | cache_rate × dataset_size | 10% of 484 subjects cached = ~48 volumes in RAM |
| Preprocessing pipeline | 2-3× per volume during transforms | Resampling creates temporary copies |
| DataLoader workers | `num_workers=4` → 4 parallel processes | Each worker holds its own copy of current batch |
| Python overhead | ~1-2 GB | PyTorch, MONAI, libraries, objects, metadata |

---

### 4.2 RAM Requirements

```
Minimum: 12-16 GB RAM
  → Can load data, but limited cache_rate=0.1 (slow I/O)
  → May need num_workers=2 instead of 4

Recommended: 32 GB RAM
  → Comfortable caching (cache_rate=0.3)
  → 4 workers, no swapping

Ideal: 64+ GB RAM
  → Cache entire dataset in memory (cache_rate=1.0, fastest training)
  → 8 workers for maximum throughput

Colab free tier:  ~12 GB RAM  → cache_rate=0.1 (10%)
Colab Pro:        ~50 GB RAM  → cache_rate=0.5 (50%)
Colab Pro+:       ~80 GB RAM  → cache_rate=0.8 (80%)
```

---

### 4.3 RAM Bottleneck Symptoms

```
If RAM is insufficient:
  → OS starts swapping to disk (thrashing)
  → Data loading becomes 10-100x slower
  → GPU sits idle waiting for data (utilization drops to 10-30%)
  → Training appears to "hang" between batches

How to detect:
  - GPU utilization < 80% → likely CPU/RAM bottleneck
  - Training step time varies wildly → data loading is inconsistent
  - System becomes unresponsive → swapping to disk

How to fix:
  1. Reduce cache_rate (use less cached data)
  2. Reduce num_workers (fewer parallel loaders)
  3. Use persistent_workers=True (avoid respawning overhead)
```

---

## 5. Storage — Dataset & Checkpoints

### 5.1 Storage Breakdown

| Item | Size | Access Pattern |
|---|---|---|
| MSD Brain Tumor dataset | ~4.5 GB compressed, ~7 GB extracted | Random access during data loading |
| BraTS 2023 dataset | ~15-20 GB | Random access |
| Model checkpoints (best.pth) | ~100 MB each | Write every best validation epoch |
| Model checkpoints (latest.pth) | ~100 MB each | Write every epoch |
| W&B logs | ~50-200 MB per experiment | Sequential write |
| Python environment (.venv) | ~3-5 GB (PyTorch, MONAI, all deps) | Read at import time |
| Predictions output | ~50-200 MB per subject (NIfTI) | Write during inference |

---

### 5.2 SSD vs HDD Impact

```
Loading a single NIfTI volume:
  HDD (spinning disk): ~200-500 ms  (mechanical seek time dominates)
  SATA SSD:            ~10-50 ms    (no mechanical parts)
  NVMe SSD:            ~1-5 ms     (PCIe direct access)

Training impact with 484 subjects, 4 workers, 100 epochs:
  HDD: data loading is the bottleneck → GPU underutilized
  SSD: data loading keeps up with GPU → full GPU utilization

Google Colab uses SSD storage — not a bottleneck.
```

---

### 5.3 Total Storage Needed

```
Minimum setup (MSD only):
  Dataset:       7 GB
  Environment:   5 GB
  Checkpoints:   1 GB
  Outputs:       2 GB
  ─────────────────────
  Total:        15 GB

Full setup (MSD + BraTS):
  Datasets:     27 GB
  Environment:   5 GB
  Checkpoints:   2 GB
  Outputs:       5 GB
  ─────────────────────
  Total:        39 GB

Colab free tier:  ~100 GB available → sufficient for everything
```

---

## 6. CPU — Preprocessing & Orchestration

### 6.1 What the CPU Does

| Task | CPU Workload |
|---|---|
| NIfTI file I/O | Read compressed .nii.gz files (gzip decompression) |
| Spatial transforms | Resampling, orientation correction, cropping (before GPU transfer) |
| Data augmentation | Random flip, rotate, intensity transforms |
| DataLoader workers | 4 parallel processes loading and preprocessing simultaneously |
| RECIST measurement | Connected component labeling, diameter computation (scipy, NumPy) |
| Hydra config parsing | YAML loading, experiment setup |
| W&B logging | HTTP requests to Weights & Biases servers |
| Checkpoint saving | Serialize model state dict to disk |

---

### 6.2 CPU Requirements

```
Cores:
  num_workers=4 → at least 4 CPU cores for data loading
  + 1 core for main training loop
  + 1 core for OS overhead
  Minimum: 4 cores
  Recommended: 8+ cores

Clock speed:
  Resampling a 240³ volume to 1mm isotropic is single-threaded per volume
  Faster clock → faster preprocessing per sample

Colab allocation:
  Free:  2 vCPUs  → may bottleneck data loading, use num_workers=2
  Pro:   4 vCPUs  → adequate for num_workers=4
  Pro+:  8 vCPUs  → comfortable headroom
```

---

## 7. Network Bandwidth — One-Time Setup

Network is only needed during initial setup, NOT during training.

| Task | Data Volume | When |
|---|---|---|
| Download MSD dataset | ~4.5 GB | Once, at setup |
| Download BraTS 2023 | ~15-20 GB | Once, at setup |
| pip install dependencies | ~3-5 GB | Once, at setup |
| Clone repo from GitHub | ~5-10 MB | Once, at setup |
| W&B logging | ~1-10 MB per epoch | During training (negligible) |
| Git push | ~1-50 MB per commit | After code changes (negligible) |

```
Minimum:  10 Mbps  (MSD download takes ~60 minutes)
Colab:    50-200 Mbps typical (MSD download takes ~3-5 minutes)

Network is NOT a bottleneck during training — only during initial setup.
```

---

## 8. Google Colab Hardware Profile

| Component | Free Tier | Pro ($10/mo) | Pro+ ($50/mo) |
|---|---|---|---|
| **GPU** | T4 (16 GB VRAM) | T4 / V100 / A100 (16-40 GB) | A100 (40-80 GB) |
| **GPU CUDA Cores** | 2,560 | 2,560-6,912 | 6,912 |
| **System RAM** | ~12 GB | ~50 GB | ~80 GB |
| **Disk** | ~100 GB SSD | ~100 GB SSD | ~200 GB SSD |
| **CPU** | 2 vCPUs | 4 vCPUs | 8 vCPUs |
| **Session time** | ~12 hours max | ~24 hours | ~24 hours |
| **Priority access** | Low (may disconnect) | Medium | High (stable sessions) |
| **OncoSeg feasibility** | Possible (tuned settings) | Good | Ideal |

---

## 9. OncoSeg Colab Configuration Guide

Settings tuned for each Colab tier:

### Free Tier (T4, 12 GB RAM)
```yaml
training:
  max_epochs: 100          # Fit in 12-hour session
  batch_size: 1            # Conservative VRAM usage
  sw_batch_size: 2         # Smaller sliding window batches
  
data:
  roi_size: [128, 128, 128]
  cache_rate: 0.1          # Limited RAM
  num_workers: 2           # Limited CPU
  
# Estimated time: ~8-10 hours for 100 epochs
```

### Pro Tier (V100, 50 GB RAM)
```yaml
training:
  max_epochs: 200          # More training within 24 hours
  batch_size: 2            # Full batch size
  sw_batch_size: 4         # Standard sliding window
  
data:
  roi_size: [128, 128, 128]
  cache_rate: 0.3          # More caching
  num_workers: 4           # Full parallel loading
  
# Estimated time: ~12-18 hours for 200 epochs
```

### Pro+ Tier (A100, 80 GB RAM)
```yaml
training:
  max_epochs: 300          # Full training schedule
  batch_size: 4            # Large batches for stable gradients
  sw_batch_size: 8         # Fast sliding window
  
data:
  roi_size: [128, 128, 128]
  cache_rate: 0.8          # Most data cached
  num_workers: 4           # Full parallel loading
  
# Estimated time: ~15-24 hours for 300 epochs
```

---

## 10. Hardware vs Data — The Real Hierarchy

```
Complete importance ranking for OncoSeg:

Priority  Component      Category    Role
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#1        DATA           Data        Defines what the model can learn
#2        Algorithm      Software    Defines how the model learns
#3        GPU VRAM       Hardware    Must hold model + activations + batch
#4        GPU Compute    Hardware    All matrix ops, convolutions, attention
#5        System RAM     Hardware    Data loading, caching, preprocessing
#6        Storage (SSD)  Hardware    Dataset + checkpoints + environment
#7        CPU Cores      Hardware    Parallel data loading workers
#8        Network        Hardware    Dataset download (one-time)
```

The relationship between data and hardware:

```
┌─────────────────────────────────────────────────────────┐
│                    DATA                                  │
│  484 brain tumor MRI scans + expert annotations         │
│  This defines the CEILING of what's achievable          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │              ALGORITHM                           │    │
│  │  OncoSeg architecture, loss functions, etc.      │    │
│  │  This determines how close we get to ceiling     │    │
│  │                                                  │    │
│  │  ┌──────────────────────────────────────────┐   │    │
│  │  │           HARDWARE                        │   │    │
│  │  │  GPU, RAM, SSD, CPU                       │   │    │
│  │  │  This determines how FAST we get there    │   │    │
│  │  └──────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘

Data sets the ceiling.
Algorithm determines accuracy.
Hardware determines speed.
```

---

*This document covers all hardware and data requirements for the OncoSeg project. Use the Colab configuration guide (Section 9) to set optimal parameters for your available tier.*
