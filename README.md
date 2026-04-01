# OncoSeg

### 3D Multi-Scale Tumor Segmentation for Automated Treatment Response Assessment

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.1+](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![MONAI 1.3+](https://img.shields.io/badge/MONAI-1.3+-green.svg)](https://monai.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Abstract

Manual tumor measurement in oncology clinical trials is slow, subjective, and limited to 2D RECIST criteria. Radiologists spend 15-30 minutes per patient per timepoint, with 20-40% inter-reader variability on tumor boundaries. This bottleneck delays drug development timelines and introduces inconsistency into imaging endpoints.

**OncoSeg** is a hybrid CNN-Transformer architecture for automated 3D tumor segmentation and treatment response assessment. Our method makes three key contributions:

1. **Hybrid Swin Transformer-CNN U-Net** — A 3D encoder-decoder architecture combining Swin Transformer's long-range attention with CNN's efficient upsampling, connected via cross-attention skip connections.
2. **Uncertainty-Aware Segmentation** — Monte Carlo Dropout-based confidence estimation that highlights ambiguous tumor boundaries for radiologist review, enabling trustworthy human-AI collaboration.
3. **Automated Response Assessment** — A downstream module that computes volumetric changes, auto-RECIST measurements, and classifies treatment response (CR/PR/SD/PD) from segmentation outputs.

We evaluate on BraTS 2023 (2,000 MRI volumes), KiTS23 (599 CT volumes), and LiTS (201 CT volumes), with systematic ablation studies quantifying the contribution of each architectural component.

## Research Questions

- **RQ1**: Does a hybrid Swin Transformer encoder with cross-attention skip connections outperform existing 3D segmentation architectures (nnU-Net, UNETR, Swin UNETR)?
- **RQ2**: How does uncertainty quantification via MC Dropout compare to ensemble methods for calibrated confidence estimation?
- **RQ3**: What is the relative contribution of the transformer encoder, cross-attention skips, deep supervision, and pretraining to segmentation performance?

## Architecture

```
Input: 3D CT/MRI Volume (H x W x D)
│
▼
┌──────────────────────────────────────────────────┐
│  Encoder: 3D Swin Transformer                     │
│  ├── Stage 1: C=48,  res=H/4   (patch embed)     │
│  ├── Stage 2: C=96,  res=H/8   (patch merge)     │
│  ├── Stage 3: C=192, res=H/16  (patch merge)     │
│  └── Stage 4: C=384, res=H/32  (bottleneck)      │
└───────────────────┬──────────────────────────────┘
                    │  Cross-Attention Skip Connections
                    ▼
┌──────────────────────────────────────────────────┐
│  Decoder: CNN Upsampling Path                     │
│  ├── Stage 4→3: TransConv3D + Cross-Attn Skip     │
│  ├── Stage 3→2: TransConv3D + Cross-Attn Skip     │
│  ├── Stage 2→1: TransConv3D + Cross-Attn Skip     │
│  └── Deep Supervision at each decoder stage       │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│  Segmentation Head                                │
│  ├── Multi-class output (tumor subregions)        │
│  ├── MC Dropout uncertainty estimation            │
│  └── Auto-RECIST measurement from predicted mask  │
└───────────────────┬──────────────────────────────┘
                    ▼
┌──────────────────────────────────────────────────┐
│  Response Assessment Module                       │
│  ├── Volumetric change: ΔV between timepoints     │
│  ├── Auto-RECIST: longest axis from mask          │
│  └── Classification: CR / PR / SD / PD            │
└──────────────────────────────────────────────────┘
```

## Datasets & Benchmarks

| Dataset | Modality | Size | Task | Role |
|---------|----------|------|------|------|
| [BraTS 2023](https://www.synapse.org/brats) | MRI | 2,000 volumes | Brain tumor (3 subregions) | Primary benchmark |
| [KiTS23](https://kits-challenge.org/kits23/) | CT | 599 volumes | Kidney tumor + cyst | Generalization test |
| [LiTS](https://competitions.codalab.org/competitions/17094) | CT | 201 volumes | Liver tumor | Small lesion evaluation |
| [BTCV](https://www.synapse.org/Synapse:syn3193805) | CT | 50 volumes | 13-organ segmentation | Multi-organ transfer |

## Evaluation Metrics

| Metric | Description | Target |
|--------|------------|--------|
| Dice Score | Volumetric overlap | ≥ 0.93 (whole tumor) |
| Hausdorff Distance 95% | Boundary accuracy (mm) | < 5.0 mm |
| Sensitivity | Missed tumor detection | > 0.92 |
| Average Surface Distance | Mean surface error (mm) | < 2.0 mm |
| Inference Time | Clinical usability | < 30s per volume |

## Baselines

| Method | BraTS Dice (WT) | Source |
|--------|----------------|--------|
| 3D U-Net | 0.88 | Cicek et al., 2016 |
| nnU-Net | 0.91 | Isensee et al., 2021 |
| UNETR | 0.89 | Hatamizadeh et al., 2022 |
| Swin UNETR | 0.92 | Tang et al., 2022 |
| **OncoSeg (Ours)** | **TBD** | — |

## Ablation Study

| # | Experiment | Tests |
|---|-----------|-------|
| A1 | Encoder: CNN vs. Transformer vs. Hybrid | Is Swin encoder worth the compute cost? |
| A2 | Skip: Concatenation vs. Cross-Attention | Does attention-based fusion improve accuracy? |
| A3 | Deep Supervision: On vs. Off | Impact on convergence and small tumor detection |
| A4 | Uncertainty: MC Dropout vs. Ensemble vs. None | Best calibration method |
| A5 | Pretraining: ImageNet-22K vs. scratch vs. SSL | Transfer learning value on medical data |
| A6 | Input resolution: 128³ vs. 192³ vs. 256³ | Resolution vs. memory tradeoff |

## Robustness Evaluation

- Cross-dataset generalization (train BraTS → test KiTS)
- Scanner variation (GE vs. Siemens vs. Philips)
- Noise injection (Gaussian, motion artifacts)
- Tumor size stratification (small < 1cm³ vs. large)
- Out-of-distribution detection

## Installation

```bash
# Clone repository
git clone https://github.com/youseihuayu-wonderful/OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment.git
cd OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e ".[all]"
```

## Quick Start

```bash
# Download BraTS dataset
python data/scripts/download_brats.py --output data/raw/brats2023

# Preprocess data
python data/scripts/preprocess.py --dataset brats --input data/raw/brats2023 --output data/processed/brats2023

# Train OncoSeg
oncoseg-train experiment=brats_oncoseg

# Train baseline (nnU-Net)
oncoseg-train experiment=brats_nnunet

# Evaluate
oncoseg-eval experiment=brats_oncoseg checkpoint=experiments/brats_oncoseg/best.pth

# Run full ablation
bash scripts/run_ablation.sh
```

## Project Structure

```
OncoSeg/
├── configs/                    # Hydra configuration files
│   ├── config.yaml             # Default config
│   ├── model/                  # Model architectures
│   ├── data/                   # Dataset configs
│   └── experiment/             # Full experiment configs
├── data/
│   ├── raw/                    # Downloaded datasets (gitignored)
│   ├── processed/              # Preprocessed volumes (gitignored)
│   └── scripts/                # Download & preprocessing scripts
├── src/
│   ├── models/                 # Model implementations
│   │   ├── oncoseg.py          # Main OncoSeg architecture
│   │   ├── baselines/          # 3D U-Net, nnU-Net, UNETR, Swin UNETR
│   │   └── modules/            # Swin blocks, cross-attention, deep supervision
│   ├── data/                   # Dataset classes, transforms, augmentation
│   ├── training/               # Trainer, losses, schedulers
│   ├── evaluation/             # Metrics, statistical tests, visualization
│   └── response/               # RECIST measurement, response classification
├── experiments/                # Saved experiment results
├── notebooks/                  # Exploratory analysis & figures
├── tests/                      # Unit & integration tests
├── paper/                      # LaTeX manuscript
│   └── figures/                # Paper figures
├── figures/                    # General figures & visualizations
├── dashboard/                  # Streamlit analytics dashboard
├── pyproject.toml              # Project configuration & dependencies
└── README.md
```

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.11 |
| DL Framework | PyTorch 2.1+ |
| Medical Imaging | MONAI 1.3+ |
| NLP (Reports) | Transformers (BioClinicalBERT, BioGPT) |
| Experiment Tracking | Weights & Biases |
| Configuration | Hydra + OmegaConf |
| Testing | pytest |
| Linting | Ruff |
| Type Checking | mypy |

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## Citation

```bibtex
@software{oncoseg2026,
  title={OncoSeg: 3D Multi-Scale Tumor Segmentation for Automated Treatment Response Assessment},
  year={2026},
  url={https://github.com/youseihuayu-wonderful/OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment}
}
```
