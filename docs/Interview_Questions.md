# OncoSeg — Interview Questions & Preparation Guide

Comprehensive list of questions a hiring manager may ask about this project, organized by category. Each question includes suggested answer points.

---

## 1. Project Overview & Motivation

### Q1: Can you walk me through your OncoSeg project at a high level?
- 3D tumor segmentation using hybrid CNN-Transformer architecture
- Targets automated treatment response assessment in oncology clinical trials
- Solves the manual tumor measurement bottleneck (RECIST)
- Benchmarked against UNet3D, UNETR, SwinUNETR on MSD Brain Tumor dataset

### Q2: What problem does this project solve, and why does it matter clinically?
- Radiologists manually measure tumors for treatment response — slow, subjective, error-prone
- RECIST 1.1 criteria require precise measurements across time points
- Automated segmentation + RECIST measurement removes human variability and speeds up clinical trials

### Q3: Why did you choose brain tumor segmentation specifically?
- BraTS/MSD Brain Tumor is a well-established benchmark with public data
- Multi-region segmentation (ET, TC, WT) is clinically meaningful
- Overlapping regions make it a challenging multi-label problem

### Q4: Who would be the end user of this system?
- Radiologists in clinical trials for treatment response assessment
- Research oncologists evaluating drug efficacy
- Clinical trial coordinators needing standardized measurements

### Q5: How does your project compare to existing solutions like nnU-Net?
- nnU-Net is self-configuring but purely CNN — no attention mechanisms
- OncoSeg adds cross-attention skip connections for better feature fusion
- MC Dropout provides uncertainty estimates — nnU-Net doesn't
- Integrated RECIST assessment pipeline goes beyond segmentation

---

## 2. Architecture & Design Decisions

### Q6: Why a hybrid CNN-Transformer architecture instead of pure CNN or pure Transformer?
- Transformers capture long-range dependencies (important for large tumors spanning many slices)
- CNNs are better at local feature extraction and are more parameter-efficient
- Hybrid gives best of both: global context from Swin Transformer encoder, fine-grained decoding from CNN decoder

### Q7: Why Swin Transformer specifically over ViT or other Transformers?
- Swin uses shifted windows — O(n) complexity vs O(n^2) for vanilla ViT
- Hierarchical feature maps (like CNN) — natural for U-Net style skip connections
- Pre-trained weights available from MONAI for medical imaging

### Q8: Explain your cross-attention skip connections. Why not simple concatenation?
- Standard U-Net concatenates encoder and decoder features at each level
- Cross-attention allows the decoder to selectively attend to relevant encoder features
- The decoder features serve as queries, encoder features as keys/values
- This filters noise and focuses on spatially relevant information
- Ablation study shows it improves Dice over simple additive skips

### Q9: How does your decoder work?
- Transposed convolution for upsampling (2x at each level)
- InstanceNorm + LeakyReLU activation
- Cross-attention fusion with encoder features at each resolution
- Final 4x upsampling to match input resolution

### Q10: Why did you choose InstanceNorm over BatchNorm?
- Batch size is 1 (3D medical images are large) — BatchNorm statistics are unreliable with batch_size=1
- InstanceNorm normalizes per-sample, per-channel — works well with any batch size
- Standard practice in medical image segmentation

### Q11: Explain your deep supervision strategy.
- Auxiliary loss heads at intermediate decoder resolutions
- Each head predicts the final segmentation at a lower resolution (upsampled to match target)
- Weighted sum: earlier (deeper) heads get exponentially decaying weights (0.5^i)
- Forces intermediate features to be semantically meaningful
- Helps with gradient flow in deep networks

### Q12: What is Monte Carlo Dropout and why did you use it?
- At inference, run multiple forward passes with dropout enabled
- Each pass gives slightly different predictions
- Mean = final prediction, variance = uncertainty estimate
- Clinically valuable: flags uncertain regions for radiologist review
- Based on Gal & Ghahramani (2016) — dropout as Bayesian approximation

### Q13: How many parameters does OncoSeg have vs the baselines?
- OncoSeg: 2.9M (with embed_dim=24) / ~12M (with embed_dim=48)
- UNet3D: ~3M
- UNETR: ~130M (ViT is parameter-heavy)
- SwinUNETR: ~12M
- OncoSeg is competitive in size while adding cross-attention and uncertainty

### Q14: Why embed_dim=24 instead of 48 for local training?
- Memory constraint on Apple M1 8GB
- 48 is the default for publication-grade results
- 24 gives directionally correct results for pipeline validation

### Q15: What is the receptive field of your model?
- Swin Transformer with window_size=(7,7,7) and 4 stages
- Shifted windows allow cross-window communication
- Effective receptive field covers the entire input volume after 4 stages
- Patch size (4,4,4) means each initial token covers 4^3 = 64 voxels

### Q16: How do you handle different input resolutions?
- Sliding window inference with configurable roi_size
- SpatialPad ensures minimum size
- Trilinear interpolation for any resolution mismatch in decoder

---

## 3. Loss Functions & Optimization

### Q17: Why did you switch from softmax to sigmoid?
- BraTS tumor regions overlap: WT contains TC, TC contains ET
- Softmax assumes mutually exclusive classes — wrong for overlapping regions
- Sigmoid treats each channel independently — correct for multi-label
- Each channel predicts: is this voxel part of TC? WT? ET?

### Q18: Explain your loss function (DiceCE).
- Combination of Dice Loss + Binary Cross-Entropy with Logits
- Dice Loss: directly optimizes the Dice metric (overlap-based)
- BCE: pixel-wise classification loss, provides stable gradients
- Equal weighting (0.5 each) balances region-level and voxel-level optimization
- Dice handles class imbalance naturally; BCE provides per-voxel signal

### Q19: Why not just use Dice Loss alone?
- Dice Loss gradients can be noisy when predictions are very wrong (early training)
- BCE provides stable per-voxel gradients that help early convergence
- Combined loss is standard in medical segmentation (nnU-Net uses it too)

### Q20: How do you handle class imbalance in tumor segmentation?
- Enhancing tumor (ET) is much smaller than whole tumor (WT)
- Dice Loss inherently handles imbalance (normalizes by region size)
- Multi-label formulation means each region is trained independently
- No explicit class weighting needed

### Q21: What optimizer do you use and why?
- AdamW with weight_decay=1e-5
- AdamW decouples weight decay from gradient updates (better regularization than L2 in Adam)
- Standard for Transformer training

### Q22: Explain your learning rate schedule.
- CosineAnnealingLR from initial lr (1e-4) to min_lr (1e-6)
- Smooth decay avoids sudden drops
- Allows the model to fine-tune in later epochs with smaller steps
- No warmup in current config (could be added)

### Q23: Why gradient clipping at max_norm=1.0?
- Transformers can have exploding gradients, especially early in training
- Clipping stabilizes training without changing gradient direction
- 1.0 is a conservative but standard choice

---

## 4. Data & Preprocessing

### Q24: How does the MSD Brain Tumor dataset work?
- 484 subjects, each with 4 MRI modalities (FLAIR, T1, T1ce, T2) in a single 4D NIfTI
- Labels: 0=background, 1=edema, 2=non-enhancing tumor, 3=enhancing tumor
- Split 80/20 for train/val (388/96)

### Q25: How do you convert MSD labels to multi-channel format?
- TC (Tumor Core) = labels 2 + 3
- WT (Whole Tumor) = labels 1 + 2 + 3
- ET (Enhancing Tumor) = label 3
- Results in 3 binary channel masks — correct for sigmoid multi-label

### Q26: What preprocessing steps do you apply?
- Orientation to RAS (standard neuroimaging convention)
- Resampling to isotropic 1mm spacing
- Z-score normalization per channel (nonzero voxels only)
- Foreground cropping to remove empty space
- Random spatial crop to roi_size during training

### Q27: What data augmentation do you use?
- Random flips (all 3 axes, p=0.5 each)
- Random 90-degree rotations (p=0.5)
- Random intensity scaling (factor=0.1, p=0.5)
- Random intensity shifting (offset=0.1, p=0.5)
- No elastic deformation (computationally expensive in 3D)

### Q28: Why normalize only nonzero voxels?
- Brain MRI has a lot of background (air = 0)
- Including zeros would skew mean/std
- Normalizing only brain voxels gives meaningful intensity statistics

### Q29: How do you handle varying volume sizes?
- CropForeground removes padding/air
- SpatialPad ensures minimum roi_size
- RandSpatialCrop extracts fixed-size patches for training
- Sliding window inference at test time handles full volumes

### Q30: Why sliding window inference instead of processing the whole volume?
- 3D volumes are too large for GPU memory (e.g., 240x240x155 with 4 channels)
- Sliding window processes roi_size patches with overlap
- Overlapping regions are averaged for smooth predictions
- Standard approach in MONAI and nnU-Net

---

## 5. Evaluation & Metrics

### Q31: What metrics do you use and why?
- **Dice Score**: volumetric overlap — the primary metric for segmentation
- **HD95**: 95th percentile Hausdorff distance — measures worst-case boundary error
- **ASD**: average surface distance — mean boundary error
- **Sensitivity**: tumor detection rate
- **Specificity**: healthy tissue preservation

### Q32: Why Dice Score instead of IoU (Jaccard)?
- Dice = 2*|A∩B|/(|A|+|B|), IoU = |A∩B|/|A∪B|
- Dice is standard in BraTS challenge and medical imaging literature
- They're monotonically related: Dice = 2*IoU/(1+IoU)
- Either works; Dice is convention

### Q33: Why HD95 instead of full Hausdorff Distance?
- Full Hausdorff is sensitive to single outlier voxels
- 95th percentile ignores the worst 5% — more robust and clinically meaningful
- A single misclassified voxel far from the tumor would dominate full Hausdorff

### Q34: Why include_background=True for your metrics?
- We switched to multi-label sigmoid — all 3 channels are foreground (TC, WT, ET)
- There is no background channel to exclude
- include_background=False would incorrectly drop channel 0 (TC)
- This was an actual bug we caught and fixed

### Q35: How do you ensure statistical significance?
- Paired Wilcoxon signed-rank tests between models (non-parametric)
- Per-subject Dice scores compared pairwise
- Alpha = 0.05 with correction for multiple comparisons

### Q36: What would you do if OncoSeg didn't outperform baselines?
- Analyze failure cases (failure_analyzer.py) — where and why it fails
- Check if cross-attention is helping (ablation study)
- Try larger embed_dim, more epochs, different augmentation
- Consider pre-trained encoder weights from MONAI

---

## 6. Treatment Response Assessment (RECIST)

### Q37: What is RECIST 1.1?
- Response Evaluation Criteria in Solid Tumors
- Standard protocol for measuring tumor response to treatment
- Measures longest axial diameter of target lesions
- Categories: CR (complete response), PR (partial response), SD (stable disease), PD (progressive disease)

### Q38: How do you automate RECIST measurement?
- Segment tumor from the predicted mask
- Find connected components (individual lesions)
- For each lesion: compute longest axial diameter and volume
- Compare baseline vs follow-up measurements
- Classify response: CR (no tumor), PR (>30% decrease), PD (>20% increase or new lesions), SD (otherwise)

### Q39: What are the limitations of your automated RECIST?
- Depends on segmentation quality — errors propagate
- Doesn't handle non-measurable disease
- Current validation is on synthetic test cases, not clinical data
- Real clinical workflow would need radiologist verification

### Q40: How would you validate RECIST in a clinical setting?
- Compare automated measurements to manual radiologist measurements
- Report concordance (correlation, Bland-Altman plots)
- Inter-observer variability as baseline for comparison
- Prospective study with clinical outcome data

---

## 7. Training Infrastructure & Engineering

### Q41: Why MONAI instead of building from scratch?
- MONAI is the standard medical imaging deep learning framework
- Provides validated transforms, losses, metrics, network components
- Sliding window inference, data loading, all optimized for medical imaging
- Built on PyTorch — familiar API, easy to extend

### Q42: Why Hydra for configuration?
- Hierarchical YAML configs — separate model, data, experiment configs
- Easy to swap components (e.g., change model with --config-name=unet3d)
- Reproducible experiments — config is logged automatically
- Supports config composition and overrides from command line

### Q43: How do you handle experiment tracking?
- Weights & Biases (W&B) integration in the trainer
- Logs loss, Dice, learning rate per epoch
- Saves model checkpoints and hyperparameters
- Enables comparison across runs

### Q44: Explain your checkpoint/resume system.
- Saves checkpoint every epoch: model weights, optimizer state, scheduler state, training history
- On restart, detects checkpoint and resumes from last completed epoch
- Completed models are skipped entirely (history saved to JSON)
- Checkpoint file deleted after successful completion
- Critical for long training runs on consumer hardware

### Q45: Why did you add MPS (Apple Silicon) support?
- Enables local development and testing without a GPU server
- MPS is Apple's Metal Performance Shaders — GPU acceleration on M1/M2
- Requires handling some MPS-specific limitations (e.g., constant padding warnings)
- Not for production training, but valuable for iteration

### Q46: How is your CI/CD set up?
- GitHub Actions: tests on Python 3.11 and 3.12
- Ruff for linting and formatting
- pytest with 46 tests covering models, losses, modules, metrics, RECIST
- Runs on every push to main

### Q47: Why num_workers=0 for data loading?
- MPS doesn't support multiprocessing well
- Avoids fork/spawn issues on macOS
- For GPU training (CUDA), would increase to 4-8

---

## 8. Coding Practices & Software Engineering

### Q48: How did you structure the codebase?
- `src/models/` — OncoSeg + baselines + modules
- `src/data/` — dataset classes + transforms
- `src/training/` — trainer + losses
- `src/evaluation/` — evaluator + metrics
- `src/response/` — RECIST + classifier
- `src/analysis/` — result analyzer + failure analyzer + figures
- `configs/` — Hydra YAML configs
- `tests/` — pytest test suite

### Q49: How do you test a deep learning project?
- Unit tests for model output shapes and gradient flow
- Tests for loss function properties (scalar output, positive, differentiable)
- Tests for RECIST with known geometric shapes (sphere volume, empty mask)
- Tests for metric computation
- Integration: full forward pass through each model variant

### Q50: What's your testing strategy for non-deterministic components?
- Fixed random seeds (42) for reproducibility
- Test properties rather than exact values (output shape, gradient existence, loss > 0)
- For RECIST: use geometric primitives with known analytical solutions

### Q51: How do you ensure code quality?
- Ruff linter + formatter (replaces flake8, black, isort)
- Type hints throughout
- Modular design — each component is independently testable
- CI enforces all checks on every push

### Q52: Why did you choose pytest over unittest?
- Cleaner syntax (plain assert statements)
- Better fixtures, parameterization, and plugin ecosystem
- Standard in modern Python projects

---

## 9. Deployment & Scalability

### Q53: How would you deploy this model in a clinical setting?
- ONNX export for inference optimization
- TorchServe or Triton Inference Server for serving
- DICOM integration for hospital PACS systems
- Uncertainty threshold to flag cases needing human review
- FDA 510(k) or De Novo pathway for regulatory clearance

### Q54: How would you scale training to larger datasets?
- Distributed Data Parallel (DDP) across multiple GPUs
- Mixed precision training (FP16) to reduce memory
- Larger batch sizes with gradient accumulation
- Pre-computed cached datasets to avoid repeated preprocessing

### Q55: How would you handle model versioning?
- W&B artifacts or MLflow model registry
- Git tags for code versions, linked to model checkpoints
- Metadata: training config, dataset version, metrics

### Q56: What would you monitor in production?
- Input data drift (intensity distribution changes)
- Prediction confidence (MC Dropout uncertainty)
- Dice score on periodic manual annotations
- Inference latency and throughput

### Q57: How would you handle edge cases in production?
- Empty predictions (no tumor found) — return clean result with confidence
- Very large tumors exceeding roi_size — sliding window handles this
- Corrupted input data — validate NIfTI headers before processing
- Out-of-distribution scans (different scanner, protocol) — uncertainty estimation flags these

---

## 10. Deep Learning Theory

### Q58: What is the vanishing gradient problem, and how does your architecture address it?
- Gradients shrink as they backpropagate through many layers
- Skip connections (from U-Net architecture) provide gradient shortcuts
- Deep supervision adds loss at intermediate layers — gradients flow directly
- LayerNorm in Transformer blocks stabilizes gradients

### Q59: What is the difference between self-attention and cross-attention?
- Self-attention: Q, K, V all come from the same input
- Cross-attention: Q from one source (decoder), K and V from another (encoder)
- In OncoSeg: decoder queries attend to encoder features, selectively fusing information

### Q60: Explain the Swin Transformer's shifted window mechanism.
- Regular windows: divide input into non-overlapping windows, compute attention within each
- Shifted windows: shift window positions by half, allowing cross-window connections
- Alternating regular/shifted windows gives global receptive field with linear complexity

### Q61: Why use learned positional embeddings vs sinusoidal?
- Swin Transformer uses relative position bias (learned, per-head)
- Relative bias captures spatial relationships between tokens
- Works for variable input sizes (unlike absolute position embeddings)

### Q62: What is the difference between Dice Loss and Cross-Entropy for segmentation?
- CE: per-voxel loss, treats each voxel independently
- Dice: region-level loss, considers the entire predicted region
- CE is dominated by the majority class (background)
- Dice naturally handles class imbalance
- Combined (DiceCE) gives both voxel-level and region-level signals

### Q63: How does dropout act as Bayesian approximation?
- Each dropout mask defines a different sub-network
- Multiple forward passes sample from the posterior distribution
- Mean prediction approximates the Bayesian model average
- Variance indicates epistemic uncertainty (model uncertainty)

---

## 11. Medical Imaging Specific

### Q64: What are the 4 MRI modalities and what does each show?
- **FLAIR**: fluid-attenuated inversion recovery — shows edema (bright)
- **T1**: structural anatomy — gray/white matter contrast
- **T1ce**: T1 with gadolinium contrast — enhancing tumor lights up
- **T2**: shows fluid and edema — complementary to FLAIR

### Q65: What is the difference between ET, TC, and WT?
- **ET (Enhancing Tumor)**: active tumor with blood-brain barrier breakdown (label 3)
- **TC (Tumor Core)**: ET + non-enhancing tumor (labels 2+3)
- **WT (Whole Tumor)**: TC + peritumoral edema (labels 1+2+3)
- They are nested/overlapping — which is why multi-label sigmoid is correct

### Q66: Why isotropic 1mm resampling?
- Different scanners produce different voxel spacings
- Resampling to uniform spacing ensures consistent model input
- 1mm is a good balance between resolution and memory

### Q67: What is the significance of RAS orientation?
- RAS = Right-Anterior-Superior — standard neuroimaging convention
- Ensures consistent spatial orientation regardless of scanner settings
- Critical for correct spatial operations (flips, rotations)

### Q68: How does your model handle multi-modal input?
- 4 MRI modalities stacked as 4 input channels (like RGB + alpha)
- The model learns to fuse information across modalities
- Each modality provides complementary information about tumor characteristics

---

## 12. Failure Analysis & Debugging

### Q69: What would you do if the model's Dice score plateaus at 0.5?
- Check label encoding (common bug: labels not properly converted)
- Verify loss function is decreasing
- Inspect a few predictions visually
- Check learning rate — may be too high or too low
- Verify data augmentation isn't too aggressive

### Q70: How do you debug a 3D medical imaging model?
- Visualize slices of input, ground truth, and prediction
- Check intermediate feature maps
- Verify data loader outputs (shapes, value ranges, label distributions)
- Start with a small subset (1-2 subjects) to overfit as sanity check

### Q71: You had a bug with include_background. How did you find and fix it?
- Training crashed with IndexError: index 2 out of bounds for size 2
- DiceMetric with include_background=False dropped channel 0, leaving only 2 values
- With 3 multi-label channels (all foreground), setting should be True
- Fixed across all files: trainer, evaluator, metrics, notebook, train_all.py

### Q72: You accidentally wiped the Colab notebook with sed. What happened?
- Used `sed -i ''` on macOS which created empty file instead of in-place edit
- Caught it because notebook parsed as empty JSON
- Restored from git history (commit 7699bdd had the intact version)
- Lesson: use proper JSON parsing for notebook edits, not sed

### Q73: What was the softmax-to-sigmoid refactor about?
- Originally used softmax (mutually exclusive classes) with 4 classes including background
- Wrong for BraTS — tumor regions overlap (WT contains TC contains ET)
- Switched to sigmoid (independent binary predictions) with 3 channels
- Changed: loss (CE → BCE), inference (argmax → threshold), metrics, all configs

---

## 13. Performance & Optimization

### Q74: How would you reduce inference time?
- ONNX export + TensorRT optimization
- Mixed precision (FP16) inference
- Reduce MC Dropout passes (trade uncertainty quality for speed)
- Optimize sliding window overlap (lower overlap = faster but less smooth)

### Q75: How would you reduce memory usage during training?
- Gradient accumulation (simulate larger batch with smaller micro-batches)
- Mixed precision training (FP16)
- Gradient checkpointing (recompute activations during backward pass)
- Smaller roi_size with more augmentation

### Q76: What is the computational complexity of your cross-attention?
- O(N_dec * N_enc * d) where N = number of spatial tokens, d = dimension
- At each decoder level, decoder tokens query all encoder tokens at that resolution
- Linear in practice because spatial resolution decreases at deeper levels

### Q77: How would you optimize the model for mobile/edge deployment?
- Knowledge distillation from OncoSeg to a smaller UNet
- Quantization (INT8)
- Pruning less important attention heads
- Not realistic for clinical use (needs server), but relevant for research

---

## 14. Research & Future Work

### Q78: What would you do differently if starting over?
- Start with sigmoid from the beginning (avoid the refactor)
- Use nnU-Net's preprocessing pipeline (proven robust)
- Add learning rate warmup
- Design for multi-GPU from the start

### Q79: How would you extend this to other tumor types?
- The architecture is tumor-agnostic — just change num_classes and data loader
- KiTS23 (kidney), LiTS (liver) configs already prepared
- Would need to adjust label conversion for each dataset's conventions

### Q80: How would you add temporal/longitudinal analysis?
- Temporal attention module already implemented
- Takes baseline and follow-up scans as input
- Cross-attention between time points to detect changes
- Enables automated treatment response tracking over time

### Q81: What is the limitation of your uncertainty estimation?
- MC Dropout only captures epistemic uncertainty (model uncertainty)
- Doesn't capture aleatoric uncertainty (inherent data noise)
- Could add heteroscedastic output (predict mean + variance per voxel)
- Number of MC passes is a trade-off: more passes = better estimate but slower

### Q82: How would you incorporate clinical metadata?
- Patient age, tumor grade, treatment history as auxiliary inputs
- Feature concatenation at bottleneck or separate MLP branch
- Multi-task learning: segmentation + survival prediction

---

## 15. Behavioral & Soft Skills

### Q83: Tell me about a time you had to debug a difficult issue in this project.
- The include_background bug: training ran for hours before crashing during validation
- Root cause: metric configuration incompatible with multi-label formulation
- Lesson: test the full pipeline (train + validate) before long runs

### Q84: How did you handle the trade-off between local training speed and accuracy?
- Reduced embed_dim (48 → 24) and roi_size for M1 feasibility
- Added checkpoint/resume to protect against crashes during multi-day training
- Designed for both local validation and Colab production training

### Q85: How do you prioritize tasks in a long-running project?
- Follow a structured plan — 28 steps tracked in progress document
- Focus on blocking items first (architecture before training, training before results)
- Use idle time productively (resume support, docs, notebook fixes while training runs)

### Q86: How do you ensure reproducibility in ML experiments?
- Fixed random seeds
- Hydra config logging
- W&B experiment tracking
- Git version control for code
- Docker/requirements for environment

### Q87: How would you explain this project to a non-technical stakeholder?
- "We built AI that automatically measures brain tumors in MRI scans"
- "It can track whether a tumor is growing or shrinking during treatment"
- "This replaces hours of manual measurement by radiologists"
- "The AI also tells you how confident it is — so doctors know when to double-check"

### Q88: What was the most challenging part of this project?
- Getting the multi-label formulation right (softmax vs sigmoid)
- Balancing model complexity with hardware constraints (M1 8GB)
- Ensuring end-to-end pipeline correctness across many components

### Q89: How do you stay current with medical imaging research?
- Follow BraTS challenge leaderboards and winner papers
- Read MICCAI, CVPR, Nature Methods publications
- Track MONAI releases and nnU-Net updates
- Arxiv for latest Transformer-based segmentation architectures

---

## 16. System Design & Architecture

### Q90: If you were designing a tumor segmentation service for a hospital, what would the architecture look like?
- PACS integration for DICOM input/output
- Preprocessing service (conversion, resampling, normalization)
- Inference service (model serving with Triton/TorchServe)
- Post-processing (connected components, RECIST measurement)
- Results viewer with uncertainty overlay
- Audit logging for regulatory compliance

### Q91: How would you handle model updates in production?
- A/B testing: new model runs in shadow mode alongside current model
- Compare predictions on incoming cases before switching
- Rollback capability if performance degrades
- Retraining pipeline triggered by data drift detection

### Q92: How would you handle PHI (Protected Health Information)?
- HIPAA compliance: encrypt data at rest and in transit
- De-identification before model training
- Federated learning to avoid centralizing patient data
- Audit trails for data access

---

## 17. Quick-Fire Technical Questions

### Q93: What's the difference between transposed convolution and bilinear upsampling?
- Transposed conv is learned — can adapt upsampling to the task
- Bilinear is fixed — faster but less flexible
- Transposed conv can cause checkerboard artifacts; careful initialization helps

### Q94: Why LeakyReLU instead of ReLU?
- ReLU kills negative activations (dead neurons)
- LeakyReLU preserves a small gradient for negatives (slope=0.01)
- More robust, especially in deeper networks

### Q95: What is the difference between InstanceNorm, BatchNorm, and LayerNorm?
- BatchNorm: normalize across batch dimension — needs large batch sizes
- InstanceNorm: normalize per sample, per channel — works with batch_size=1
- LayerNorm: normalize across feature dimension — standard in Transformers

### Q96: What is the MONAI sliding_window_inference doing?
- Splits large volume into overlapping patches of roi_size
- Runs model on each patch independently
- Averages overlapping predictions (Gaussian weighting optional)
- Returns full-volume prediction without OOM errors

### Q97: What happens if two tumor regions are predicted with different thresholds?
- Current approach: fixed 0.5 threshold for all channels
- Could optimize per-channel thresholds on validation set
- Post-processing: enforce TC ⊂ WT constraint (if voxel is TC, it must be WT)

### Q98: How does AdamW differ from Adam?
- Adam applies weight decay to the gradient before the adaptive step — incorrect
- AdamW decouples weight decay — applies it directly to weights after the step
- Mathematically: AdamW = Adam + correct L2 regularization
- Better generalization in practice, especially with Transformers

### Q99: What is the purpose of smooth_nr and smooth_dr in Dice Loss?
- Prevents division by zero when prediction or target is empty
- smooth_nr (numerator) and smooth_dr (denominator) add small constants
- Typical value: 1e-5
- Ensures stable gradients even for empty regions

### Q100: Why did you choose a patch size of (4,4,4) for the Swin Transformer?
- Each patch becomes one token — smaller patches = more tokens = higher resolution but more compute
- (4,4,4) = 64 voxels per token — good balance
- With input roi_size=96, gives 24^3 = 13,824 initial tokens (before window partitioning)
- Matches MONAI SwinUNETR default configuration
