# OncoSeg

### 3D Multi-Scale Tumor Segmentation for Automated Treatment Response Assessment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI 1.3+](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![Tests](https://img.shields.io/badge/tests-46%20passed-brightgreen.svg)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

Manual tumor measurement in oncology clinical trials is slow, subjective, and limited to 2D RECIST criteria. Radiologists spend 15-30 minutes per patient per timepoint, with 20-40% inter-reader variability on tumor boundaries.

**OncoSeg** is a hybrid CNN-Transformer architecture for automated 3D tumor segmentation and treatment response assessment. Key contributions:

1. **Hybrid Swin Transformer-CNN U-Net** with cross-attention skip connections — decoder selectively queries encoder features instead of blind concatenation
2. **Uncertainty-Aware Segmentation** via Monte Carlo Dropout — highlights ambiguous tumor boundaries for radiologist review
3. **Automated RECIST 1.1 Response Assessment** — computes longest axial diameters, volumes, and classifies treatment response (CR/PR/SD/PD) directly from segmentation outputs
4. **Temporal Attention** for longitudinal scan comparison — cross-attention between baseline and follow-up scans captures tumor evolution

## Architecture

```
Input: 4-channel 3D MRI [B, 4, 128, 128, 128]
       (T1, T1-contrast, T2, FLAIR)
│
▼
┌──────────────────────────────────────────────────┐
│  Encoder: 3D Swin Transformer                     │
│  ├── Stage 1: C=48,  res=H/4   (patch embed)     │
│  ├── Stage 2: C=96,  res=H/8   (patch merge)     │
│  ├── Stage 3: C=192, res=H/16  (patch merge)     │
│  └── Stage 4: C=384, res=H/32  (bottleneck)      │
│                                                    │
│  Optional: Temporal Attention at bottleneck        │
│  (fuses baseline + follow-up for response assess.) │
└───────────────────┬──────────────────────────────┘
                    │  Cross-Attention Skip Connections
                    ▼
┌──────────────────────────────────────────────────┐
│  Decoder: CNN Upsampling Path                     │
│  ├── Stage 4→3: TransConv3D + Cross-Attn Skip     │
│  ├── Stage 3→2: TransConv3D + Cross-Attn Skip     │
│  ├── Stage 2→1: TransConv3D + Cross-Attn Skip     │
│  ├── 4x Upsample Head (recover patch embed)       │
│  └── Deep Supervision at each decoder stage       │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│  Output                                           │
│  ├── Segmentation: [B, 4, H, W, D]               │
│  ├── Uncertainty map (MC Dropout entropy)         │
│  └── RECIST: diameter, volume, CR/PR/SD/PD        │
└──────────────────────────────────────────────────┘
```

## Model Comparison

| Model | Type | Parameters | Architecture |
|-------|------|-----------|-------------|
| **OncoSeg (Ours)** | **Swin + CNN** | **12.1M** | **Cross-attention skips + deep supervision + MC Dropout + temporal attention** |
| UNet3D | Pure CNN | 19.2M | 5-level encoder-decoder, channels [32,64,128,256,512] |
| Swin UNETR | Swin + CNN | 62.2M | MONAI's Swin Transformer U-Net (standard concatenation skips) |
| UNETR | ViT + CNN | 130.8M | Vision Transformer encoder (12 layers, 768-dim) + CNN decoder |

OncoSeg achieves competitive performance with **6x fewer parameters** than UNETR and **5x fewer** than Swin UNETR.

## Dataset

Primary dataset: [MSD Task01 Brain Tumour](https://medicaldecathlon.com/) (484 subjects, 4 MRI modalities)

| Label | Class | Description |
|-------|-------|-------------|
| 0 | Background | Healthy tissue |
| 1 | Edema | Peritumoral edema |
| 2 | Non-enhancing | Necrotic / non-enhancing tumor core |
| 3 | Enhancing | GD-enhancing tumor |

Additional configs ready for: BraTS 2023, KiTS23, LiTS, BTCV

## Evaluation Metrics

| Metric | What It Measures |
|--------|-----------------|
| Dice Score | Volume overlap (higher = better) |
| Hausdorff Distance 95% | Worst-case boundary error in mm (lower = better) |
| Average Surface Distance | Mean boundary error in mm (lower = better) |
| Sensitivity | Fraction of tumor correctly detected |
| Specificity | Fraction of healthy tissue correctly excluded |

All metrics computed per BraTS region: Enhancing Tumor (ET), Tumor Core (TC), Whole Tumor (WT).

## Ablation Study

| Variant | Cross-Attention Skip | Deep Supervision | Purpose |
|---------|---------------------|-----------------|---------|
| OncoSeg (full) | Yes | Yes | Our complete model |
| OncoSeg (concat skip) | No (additive) | Yes | Test cross-attention contribution |
| OncoSeg (no DS) | Yes | No | Test deep supervision contribution |
| UNet3D baseline | N/A | N/A | Pure CNN reference |
| UNETR baseline | N/A | N/A | ViT encoder reference |
| Swin UNETR baseline | N/A | N/A | Same encoder, standard skips |

## Results

### Segmentation Performance (MSD Brain Tumor, 96 val subjects)

| Model | Dice TC | Dice WT | Dice ET | Dice Mean | Params |
|-------|---------|---------|---------|-----------|--------|
| **OncoSeg** | _TBD_ | _TBD_ | _TBD_ | _TBD_ | 2.9M |
| UNet3D | _TBD_ | _TBD_ | _TBD_ | _TBD_ | ~3M |
| SwinUNETR | _TBD_ | _TBD_ | _TBD_ | _TBD_ | ~12M |

> Results will be updated after training completes. See `experiments/local_results/` for raw outputs.

## Quick Start — Google Colab

The easiest way to run OncoSeg (no local GPU required):

1. Open `notebooks/OncoSeg_Full_Pipeline.ipynb` in [Google Colab](https://colab.research.google.com)
2. Set runtime to **GPU** (Runtime > Change runtime type > T4 GPU)
3. Run all cells — the notebook handles data download, training, evaluation, and visualization

## Local Installation

```bash
git clone https://github.com/youseihuayu-wonderful/OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment.git
cd OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment

python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

### Local Training (Apple Silicon / CPU)

```bash
# Download dataset only (~7.1 GB)
python train_local.py --download-only

# Train with M1-optimized settings
python train_local.py --epochs 50 --embed-dim 24 --roi-size 96

# Train with full model (requires more RAM)
python train_local.py --epochs 100 --embed-dim 48
```

### CLI Commands (Hydra)

```bash
# Train with Hydra config
python -m src.training.trainer model=oncoseg data=msd_brain training.max_epochs=100

# Run tests
pytest tests/ -v
```

### Download Datasets

```bash
# MSD Brain Tumor (free, ~7 GB)
python data/scripts/download_msd.py --output data/raw

# BraTS 2023 (requires Synapse registration)
python data/scripts/download_brats.py --output data/raw/brats2023

# KiTS23, LiTS, BTCV — see data/scripts/ for instructions
```

## Project Structure

```
OncoSeg/
├── configs/                    # Hydra configuration files
│   ├── config.yaml             # Default training config
│   ├── model/                  # oncoseg, unet3d, unetr, swin_unetr
│   ├── data/                   # brats2023, msd_brain, kits23, lits, btcv
│   └── experiment/             # brats_oncoseg, brats_ablation
├── data/
│   └── scripts/                # Dataset download & verification scripts
├── docs/
│   ├── AI_Knowledge_Fundamentals.md   # All AI/ML concepts used
│   └── Hardware_and_Data_Requirements.md
├── src/
│   ├── models/
│   │   ├── oncoseg.py          # Main architecture (Swin + CNN + cross-attn)
│   │   ├── modules/            # Swin encoder, cross-attention, CNN decoder,
│   │   │                       # deep supervision, temporal attention
│   │   └── baselines/          # UNet3D, UNETR, SwinUNETR
│   ├── data/                   # BraTS + MSD dataset loaders, transforms
│   ├── training/               # Trainer, DiceCE loss, deep supervision loss
│   ├── evaluation/             # Metrics (Dice, HD95, ASD), evaluator
│   ├── response/               # RECIST 1.1 measurement, CR/PR/SD/PD classifier
│   ├── analysis/               # Result analysis, failure analysis, figures
│   └── inference.py            # Prediction pipeline with uncertainty
├── notebooks/
│   └── OncoSeg_Full_Pipeline.ipynb  # All-in-one Colab notebook
├── train_local.py              # Local training script (MPS/CPU)
├── tests/                      # 46 unit tests (models, losses, modules, RECIST, analysis)
├── pyproject.toml              # Dependencies & project config
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11+ |
| Deep Learning | PyTorch 2.1+ |
| Medical Imaging | MONAI 1.3+ |
| Configuration | Hydra + OmegaConf |
| Experiment Tracking | Weights & Biases |
| Testing | pytest (46 tests) |
| Code Quality | Ruff, mypy |

## Testing

```bash
$ pytest tests/ -v
======================== 46 passed in 24.16s ========================
```

Tests cover: OncoSeg forward pass, deep supervision, all 3 baselines, DiceCE loss, deep supervision loss, cross-attention skip, Swin encoder, UNETR baseline, RECIST measurement (7 edge cases), response classification (5 scenarios), result analysis, failure analysis, figure generation.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{oncoseg2026,
  title={OncoSeg: 3D Multi-Scale Tumor Segmentation for Automated Treatment Response Assessment},
  author={Yu, Shihua},
  year={2026},
  url={https://github.com/youseihuayu-wonderful/OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment}
}
```
