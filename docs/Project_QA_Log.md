# OncoSeg Project — Complete Q&A Coverage

Every question below covers a specific part of the project. Organized by what was actually built, step by step.

---

## 1. Project Structure & Setup

**Q: How is the project organized?**
> Modular Python package: `src/models/`, `src/data/`, `src/training/`, `src/evaluation/`, `src/response/`, `src/analysis/`. Hydra YAML configs under `configs/`. Tests under `tests/`. Entry points: `train_local.py`, `train_all.py`, `src/inference.py`.

**Q: Why use pyproject.toml instead of setup.py?**
> Modern Python packaging standard (PEP 621). Declares all dependencies, build backend (setuptools), and tool configs (pytest, ruff) in one file. CI installs with `pip install -e ".[all]"`.

**Q: Why Hydra for configuration management?**
> Hierarchical YAML configs for model, data, and experiment settings. Swap models with `--config-name=unet3d`. Configs are logged automatically for reproducibility. Supports command-line overrides.

---

## 2. MSD Brain Tumor Dataset

**Q: What is the MSD Brain Tumor dataset?**
> Medical Segmentation Decathlon Task01. 484 subjects, each with 4 MRI modalities (FLAIR, T1, T1ce, T2) stored as a single 4D NIfTI file. Labels: 0=background, 1=edema, 2=non-enhancing tumor, 3=enhancing tumor.

**Q: How do you load a 4D NIfTI file with 4 modalities?**
> `LoadImaged` from MONAI reads the file. `EnsureChannelFirstd` places channels first. The 4 modalities become 4 input channels to the model, similar to RGB+alpha.

**Q: How do you convert MSD labels into multi-channel format?**
> Custom `ConvertMSDToMultiChanneld` transform:
> - TC (Tumor Core) = labels 2 + 3
> - WT (Whole Tumor) = labels 1 + 2 + 3
> - ET (Enhancing Tumor) = label 3
> Result: 3 binary channel masks for multi-label sigmoid training.

**Q: Why 3 channels and not 4 (with background)?**
> Background is implicit — any voxel not predicted as TC/WT/ET is background. Sigmoid multi-label doesn't need an explicit background channel. This also reduces computation.

---

## 3. BraTS Dataset Support

**Q: How does BraTS differ from MSD in data format?**
> BraTS stores each modality as a separate NIfTI file (4 files per subject). MSD stores all 4 modalities in one 4D file. Different loaders handle each format.

**Q: What does `brats_dataset.py` do differently from `msd_dataset.py`?**
> BraTS loader reads 4 separate files and stacks them. MSD loader reads one 4D file. Both produce the same output: [4, H, W, D] image tensor + [3, H, W, D] label tensor.

---

## 4. Preprocessing Transforms

**Q: What preprocessing pipeline do you apply?**
> 1. Load NIfTI → 2. Ensure channel-first → 3. Convert labels to multi-channel → 4. Orient to RAS → 5. Resample to 1mm isotropic → 6. Z-score normalize (nonzero voxels, per channel) → 7. Crop foreground → 8. Pad to minimum roi_size → 9. Random crop to roi_size.

**Q: Why normalize only nonzero voxels?**
> Brain MRI has large background regions (air = 0). Including zeros would skew the mean and standard deviation. Normalizing only brain tissue voxels gives meaningful intensity statistics.

**Q: What augmentations do you use during training?**
> Random flips (all 3 axes), random 90-degree rotations, random intensity scaling (±10%), random intensity shifting (±10%). No elastic deformation — too expensive in 3D.

**Q: Why orient to RAS before anything else?**
> RAS (Right-Anterior-Superior) is the standard neuroimaging convention. Different scanners may store data in different orientations. Standardizing ensures consistent spatial operations.

---

## 5. OncoSeg Model Architecture

**Q: What is the OncoSeg encoder?**
> MONAI's SwinTransformer with patch_size=(4,4,4), shifted windows of size (7,7,7). 4 stages with depths (2,2,2,2) and increasing channel dimensions (embed_dim * 1, 2, 4, 8). Outputs hierarchical feature maps at 4 resolutions.

**Q: What is the OncoSeg decoder?**
> CNN decoder with transposed convolutions for 2x upsampling at each level. InstanceNorm + LeakyReLU. Cross-attention skip connections fuse encoder features. Final 4x transposed conv restores original resolution.

**Q: How do cross-attention skip connections work?**
> At each decoder level:
> - Decoder features → Query (Q)
> - Encoder features → Key (K) and Value (V)
> - Multi-head attention: decoder selectively attends to relevant encoder features
> - Followed by FFN with residual connection and LayerNorm
> This is more selective than simple concatenation — filters noise from encoder.

**Q: How is the cross-attention implemented in code?**
> `CrossAttentionSkip` module: flatten 3D features to sequences, apply LayerNorm, project Q/K/V, compute scaled dot-product attention with multiple heads, reshape back to 3D. Includes residual connection and FFN.

**Q: What is the MC Dropout layer and where is it placed?**
> `nn.Dropout3d(p=0.1)` applied to the bottleneck (deepest encoder features). During inference with `predict_with_uncertainty()`, dropout stays enabled across N forward passes. Variance of predictions = uncertainty map.

**Q: What does deep supervision do in OncoSeg?**
> Auxiliary `nn.Conv3d(dim, num_classes, 1)` heads at each intermediate decoder resolution. Each head's output is upsampled to full resolution. Loss is computed at all scales with exponentially decaying weights. Only active during training.

---

## 6. Swin Transformer Encoder Details

**Q: What does `SwinEncoder3D` wrap?**
> MONAI's `SwinTransformer` for 3D inputs. Takes (B, C, H, W, D) input, outputs list of feature maps at decreasing resolutions. Each stage applies shifted-window self-attention.

**Q: Why shifted windows instead of global attention?**
> Global attention is O(n²) — prohibitive for 3D volumes. Shifted windows compute attention within local windows (O(n)), then shift window positions to enable cross-window communication. Linear complexity with global receptive field.

---

## 7. Baseline Models

**Q: What are the three baselines and why were they chosen?**
> - **UNet3D**: Pure CNN — tests if attention mechanisms add value
> - **UNETR**: ViT encoder — tests pure Transformer approach
> - **SwinUNETR**: Same Swin encoder as OncoSeg but with standard skip connections — isolates the contribution of cross-attention skips

**Q: How are baselines implemented?**
> Thin wrappers around MONAI's `UNet`, `UNETR`, and `SwinUNETR`. Each wrapper standardizes the interface: `__init__(in_channels, num_classes, ...)` and `forward(x) → {"pred": tensor}`.

---

## 8. Loss Functions

**Q: What loss function does OncoSeg use?**
> `DiceCELoss` = 0.5 * DiceLoss(sigmoid=True) + 0.5 * BCEWithLogitsLoss. Dice handles class imbalance; BCE provides stable per-voxel gradients.

**Q: Why BCEWithLogitsLoss instead of CrossEntropyLoss?**
> Multi-label sigmoid formulation — each channel is an independent binary prediction. BCE operates on each channel separately. CrossEntropy assumes mutually exclusive classes (softmax) which is wrong for overlapping tumor regions.

**Q: How does deep supervision loss work?**
> `DeepSupervisionLoss` wraps the base DiceCELoss. Takes a list of predictions at different scales. Applies exponentially decaying weights: w_i = 0.5^i / sum(weights). Normalized so weights sum to 1.

**Q: What are smooth_nr and smooth_dr in Dice Loss?**
> Small constants (1e-5) added to numerator and denominator of Dice formula. Prevents division by zero when a region is empty (no tumor). Ensures stable gradients.

---

## 9. Training Pipeline

**Q: What does the Trainer class do?**
> Full training loop: loads model and data from Hydra config, runs AdamW optimizer with CosineAnnealingLR scheduler, trains with sliding window validation, logs to W&B, saves best checkpoint by Dice score.

**Q: Why AdamW over Adam?**
> AdamW correctly decouples weight decay from gradient updates. In Adam, weight decay interacts with adaptive learning rates incorrectly. AdamW gives better generalization, especially for Transformers.

**Q: Why CosineAnnealingLR?**
> Smooth learning rate decay from 1e-4 to 1e-6 over training. No sudden drops. Allows fine-tuning in later epochs. Simpler and more predictable than step-based schedules.

**Q: Why gradient clipping at max_norm=1.0?**
> Transformers can produce large gradients, especially early in training. Clipping caps the gradient norm without changing direction. Stabilizes training.

**Q: How does sliding window validation work?**
> Full volumes are too large for GPU memory. MONAI's `sliding_window_inference` splits volume into overlapping roi_size patches, runs the model on each, and averages overlapping regions. Overlap=0.25 for validation, 0.5 for testing.

---

## 10. train_local.py

**Q: What is train_local.py for?**
> Simplified training script for Apple Silicon (MPS) with M1-safe settings: smaller embed_dim, reduced roi_size, num_workers=0. Trains OncoSeg only on MSD data.

**Q: Why num_workers=0?**
> MPS on macOS has issues with multiprocessing data loading. Setting workers to 0 avoids fork/spawn crashes. On CUDA, you would use 4-8 workers.

---

## 11. train_all.py

**Q: What does train_all.py do?**
> Self-contained script that trains all 3 models (OncoSeg, UNet3D, SwinUNETR) sequentially on MSD data. Generates training curves, saves results JSON. Optimized for M1 8GB.

**Q: Why is it self-contained (doesn't import from src/)?**
> Avoids dependency on the full package installation. Can be run standalone. Contains its own OncoSeg implementation, loss functions, and data loading. Useful for quick experiments.

**Q: How does the model factory work?**
> `build_model(name, roi_size, embed_dim)` returns the requested model. "oncoseg" → OncoSeg, "unet3d" → MONAI UNet, "swin_unetr" → MONAI SwinUNETR. All configured with NUM_CLASSES=3.

**Q: How does checkpoint/resume work?**
> Every epoch saves to `{model}_checkpoint.pth`: model weights, optimizer state, scheduler state, training history. On restart, loads checkpoint and resumes from last epoch. Completed models (no checkpoint, has best.pth + history.json) are skipped entirely. Checkpoint deleted after successful completion.

**Q: Why save optimizer and scheduler state in the checkpoint?**
> Without them, resuming would reset learning rate and momentum. AdamW maintains per-parameter running averages (m, v). CosineAnnealing needs to know which epoch it's on. Saving state ensures seamless resume.

---

## 12. Evaluation Pipeline

**Q: What does the Evaluator class do?**
> Loads a trained model and test data. Runs sliding window inference on each subject. Computes metrics via SegmentationMetrics. Supports multi-seed evaluation for variance estimation. Saves results to JSON.

**Q: What metrics does SegmentationMetrics compute?**
> - Dice Score (DiceMetric) — volumetric overlap
> - HD95 (HausdorffDistanceMetric, 95th percentile) — worst-case boundary error
> - ASD (SurfaceDistanceMetric, symmetric) — mean boundary error
> - Sensitivity and Specificity (ConfusionMatrixMetric)
> All computed per region (TC, WT, ET) with include_background=True.

**Q: Why include_background=True?**
> With multi-label sigmoid (3 channels: TC, WT, ET), all channels are foreground. include_background=False would drop channel 0 (TC), giving only 2 values instead of 3. This was a real bug that caused IndexError during validation.

**Q: Why HD95 instead of full Hausdorff?**
> Full Hausdorff is sensitive to single outlier voxels. One misclassified voxel far from the tumor would dominate. 95th percentile ignores the worst 5% — more robust and clinically meaningful.

---

## 13. RECIST Measurement

**Q: What is RECIST 1.1?**
> Response Evaluation Criteria in Solid Tumors. Standard protocol for measuring tumor response. Measures longest axial diameter of target lesions. Used in clinical trials to determine if treatment is working.

**Q: How does `recist.py` work?**
> Takes a binary segmentation mask. Finds connected components (individual lesions). For each lesion: computes longest axial diameter (max distance across axial slices) and volume (voxel count * voxel spacing). Returns measurements sorted by size.

**Q: How does `classifier.py` determine treatment response?**
> Compares baseline and follow-up RECIST measurements:
> - CR (Complete Response): no tumor in follow-up
> - PR (Partial Response): >30% decrease in sum of diameters
> - PD (Progressive Disease): >20% increase or new lesions
> - SD (Stable Disease): neither PR nor PD

**Q: How did you test RECIST with no clinical data?**
> Synthetic geometric test cases: empty mask (0 diameter), single voxel, sphere (known volume = 4/3πr³), cube (known volume). Verified against analytical solutions. Also tested CR/PR/SD/PD classification logic.

---

## 14. Inference Script

**Q: What does the Predictor class do?**
> Loads a trained model checkpoint. Runs sliding window inference on a NIfTI input. Supports standard prediction and MC Dropout uncertainty estimation. Outputs segmentation mask as NIfTI + RECIST measurements.

**Q: How does MC Dropout uncertainty work at inference?**
> Run N forward passes (default 10) with dropout enabled. Each pass gives slightly different predictions. Stack all predictions → mean = final prediction, variance = uncertainty map. High variance = model is uncertain.

**Q: What is the output format?**
> Dictionary with: "segmentation" (uint8 numpy array), "probabilities" (float32), "uncertainty" (float32, if MC Dropout), "recist" (diameter and volume measurements).

---

## 15. Analysis Toolkit

**Q: What does result_analyzer.py do?**
> Loads evaluation results from JSON. Computes: best Dice summary, comparison table across models, per-region breakdown, convergence analysis, statistical significance tests (Wilcoxon signed-rank).

**Q: What does failure_analyzer.py do?**
> Identifies subjects where the model performed worst. Stratifies by tumor size (small/medium/large). Detects segmentation biases (over/under-segmentation). Categorizes failure types.

**Q: What does figures.py generate?**
> Training loss curves, Dice comparison bar charts, ablation study charts. Publication-ready matplotlib figures with consistent styling.

---

## 16. Temporal Attention

**Q: What is temporal attention and what does it do?**
> Module for comparing baseline and follow-up scans. Takes two encoder feature sequences and computes cross-attention between time points. Detects what changed between scans.

**Q: How is it integrated into OncoSeg?**
> OncoSeg accepts optional `x_followup` input. When provided, temporal attention is applied to encoder features. Enables automated longitudinal treatment tracking.

**Q: Is temporal attention used in current training?**
> Not yet — requires paired baseline/follow-up data. The module is implemented and integrated but not exercised in the MSD single-timepoint pipeline.

---

## 17. Test Suite

**Q: What do the 46 tests cover?**
> - `test_models.py` (7): OncoSeg output shape, deep supervision, UNet3D, SwinUNETR, RECIST empty/sphere/classifier
> - `test_modules.py` (6): CrossAttentionSkip shape/gradient, DeepSupervisionHead, SwinEncoder, UNETR
> - `test_losses.py` (6): DiceCELoss scalar/positive/perfect/gradient, DeepSupervisionLoss weighted/single
> - `test_analysis.py` (15): ResultAnalyzer, FailureAnalyzer, FigureGenerator
> - `test_response.py` (11): RECIST measurer, ResponseClassifier

**Q: How do you test non-deterministic deep learning components?**
> Test properties, not exact values. Check output shapes, gradient existence (loss.backward doesn't crash), loss is positive, loss is low for perfect prediction. Fixed seeds (42) for reproducibility.

**Q: Why test gradient flow?**
> Ensures loss connects to all model parameters through the computation graph. A broken gradient path means the model won't learn. `loss.backward()` + check `param.grad is not None`.

---

## 18. CI/CD

**Q: What does the GitHub Actions pipeline do?**
> Runs on push to main. Installs package on Python 3.11 and 3.12. Runs `ruff check` for linting, `ruff format --check` for formatting, `pytest` for all 46 tests.

**Q: Why test on both Python 3.11 and 3.12?**
> Ensures compatibility across supported versions. Catches version-specific issues (e.g., type hint syntax changes, deprecated APIs).

**Q: What code quality tools do you use?**
> Ruff (replaces flake8, black, isort in one tool). Configured in pyproject.toml. Enforces consistent style, import ordering, and catches common bugs.

---

## 19. Dataset Download Scripts

**Q: What download scripts exist?**
> `download_msd.py` (MSD Brain Tumor), `download_kits23.py`, `download_lits.py`, `download_btcv.py`. Each downloads from the official source, extracts, and verifies the dataset structure.

---

## 20. Multi-Label Sigmoid Refactor

**Q: Why did you switch from softmax to sigmoid?**
> BraTS tumor regions overlap: WT ⊃ TC ⊃ ET. Softmax assumes mutually exclusive classes — incorrect. Sigmoid treats each channel as independent binary prediction — correct for overlapping regions.

**Q: What files changed in the refactor?**
> All model configs (num_classes 4→3), all model defaults, losses (CE→BCE, softmax→sigmoid in DiceLoss), trainer, evaluator, inference, train_local.py, train_all.py, notebook. 15 files total.

**Q: What bug did the refactor introduce?**
> `include_background=False` in DiceMetric dropped channel 0 (TC), leaving only 2 values. Accessing `scores[2]` caused IndexError. Fixed by setting `include_background=True` since all 3 channels are foreground.

---

## 21. Colab Notebook

**Q: What does the Colab notebook contain?**
> 44 cells in a self-contained pipeline: install dependencies → download MSD data → define model/loss → train OncoSeg + 3 baselines → evaluate with metrics → ablation study → RECIST demo → visualization → statistical tests → save results.

**Q: Why a single notebook instead of scripts?**
> Colab provides free T4 GPU. Single notebook means no installation or setup — just open and run. Self-contained: includes all model code, not just imports. Ideal for reproducibility and sharing.

**Q: What was the notebook corruption incident?**
> A macOS `sed -i ''` command accidentally emptied the notebook file. It was committed empty. Restored from git history (commit 7699bdd) and fixes were applied properly using Python JSON parsing instead of sed.

---

## 22. Documentation

**Q: What documentation exists?**
> - `README.md`: Project overview, architecture, results table, quickstart, installation
> - `docs/Paper_Methods_Draft.md`: 200+ line methods section with references, results templates
> - `docs/Pipeline_Document.md`: 1300+ line detailed technical document
> - `docs/AI_Knowledge_Fundamentals.md`: 1500+ line comprehensive AI reference
> - `docs/Interview_Questions.md`: 100 interview questions with answer points
> - `docs/Project_QA_Log.md`: This file

---

## 23. Model Profiler

**Q: What does model_profiler.py do?**
> Instantiates all models, counts parameters, measures forward pass time, and reports memory usage. Generates a comparison table showing OncoSeg's efficiency vs baselines.

---

## 24. Apple Silicon MPS Support

**Q: How did you add MPS support?**
> Auto-detection: `torch.backends.mps.is_available()`. Device selection cascade: CUDA → MPS → CPU. Added to trainer, evaluator, inference, train_local.py, train_all.py. Handles MPS-specific warnings (constant padding in 3D).

**Q: What are MPS limitations?**
> Slower than CUDA for 3D convolutions. Some ops fall back to CPU. No multiprocessing data loading (num_workers=0). Memory limited (8GB shared with system). Not for production training — useful for development.

---

## 25. Checkpoint & Resume System

**Q: Why was resume support added?**
> Training on M1 takes days (50 epochs × 388 samples × 3 models). If machine sleeps or process crashes, all progress would be lost. Checkpoint/resume saves every epoch.

**Q: What is saved in each checkpoint?**
> Model weights (`model_state_dict`), optimizer state (`optimizer_state_dict`), scheduler state (`scheduler_state_dict`), full training history (losses, Dice scores, best epoch).

**Q: How are completed models handled on restart?**
> If `{model}_best.pth` exists and `{model}_checkpoint.pth` doesn't, the model finished. Its history is loaded from `{model}_history.json` and training is skipped.

---

## 26. Results & Figures

**Q: What results are generated after training?**
> - `results.json`: config + per-model best Dice, best epoch, final loss
> - `training_curves.png`: loss and Dice plots for all models
> - `{model}_best.pth`: best model weights by Dice score
> - Per-model history JSON files

**Q: What does the results table look like?**
> | Model | Dice TC | Dice WT | Dice ET | Dice Mean | Params |
> Populated from `results.json` after training completes.

---

## 27. Key Technical Decisions Summary

| Decision | What | Why |
|----------|------|-----|
| Sigmoid over softmax | Multi-label formulation | Tumor regions overlap (WT ⊃ TC ⊃ ET) |
| 3 classes, no background | num_classes=3 | Background is implicit in sigmoid |
| DiceCE loss | Dice + BCE | Class imbalance (Dice) + stable gradients (BCE) |
| include_background=True | Metrics config | All 3 channels are foreground in multi-label |
| Cross-attention skips | Encoder-decoder fusion | Selective attention > simple concatenation |
| MC Dropout at bottleneck | Uncertainty estimation | Clinical need to flag uncertain predictions |
| InstanceNorm | Normalization | Batch size=1 makes BatchNorm unreliable |
| Sliding window inference | Full-volume prediction | 3D volumes too large for GPU memory |
| Checkpoint every epoch | Resume support | Multi-day training on consumer hardware |
| Hydra configs | Configuration | Reproducibility + easy experiment switching |
