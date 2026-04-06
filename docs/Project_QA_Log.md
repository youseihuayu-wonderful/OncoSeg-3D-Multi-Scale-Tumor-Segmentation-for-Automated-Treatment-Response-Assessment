# OncoSeg — Interview Questions: How the Project Was Built

Questions a hiring manager would ask about how you designed, built, and iterated on this project. Answer in first person — tell the story of building it.

---

## Getting Started

**Q1: What made you decide to build this project?**
> Tumor measurement in clinical trials is a huge bottleneck — radiologists manually measure lesions using RECIST criteria, which is slow, subjective, and error-prone. I wanted to automate both the segmentation and the response assessment in one pipeline.

**Q2: How did you decide on the tech stack?**
> PyTorch was a given for deep learning. I chose MONAI because it's the standard for medical imaging — provides validated transforms, losses, and network components. Hydra for config management because I needed to swap models and datasets easily. W&B for experiment tracking.

**Q3: How did you plan the project architecture before writing any code?**
> I broke it into clear phases: (1) project structure and configs, (2) data loading, (3) model architecture, (4) training pipeline, (5) evaluation, (6) RECIST assessment, (7) testing, (8) notebook for reproducibility. Each phase builds on the previous one.

**Q4: Why did you start with project structure before any ML code?**
> Having the right structure from the start saves refactoring later. I set up pyproject.toml, Hydra configs, the src/ package layout, and the test directory first. This meant when I wrote the model code, it already had a home and a config to go with it.

---

## Data Pipeline

**Q5: Walk me through how you built the data pipeline.**
> First I studied the MSD dataset format — 4D NIfTI files with 4 MRI modalities. I wrote a custom dataset class that reads these files and a transform to convert the integer labels (0,1,2,3) into multi-channel binary masks (TC, WT, ET). Then I added BraTS support separately because it stores modalities as separate files.

**Q6: What was the trickiest part of the preprocessing?**
> Getting the label conversion right. MSD labels are integers {0,1,2,3} but the tumor regions overlap — whole tumor contains tumor core which contains enhancing tumor. I had to convert to 3 binary channels where each channel independently says "is this voxel part of this region?" This directly drove the decision to use sigmoid instead of softmax.

**Q7: How did you decide on the augmentation strategy?**
> I kept it simple — random flips, rotations, and intensity variations. I avoided elastic deformation because it's computationally expensive in 3D and the benefit is marginal for brain tumors. The augmentations I chose are fast and address the main sources of variation in MRI data.

---

## Model Architecture

**Q8: Walk me through your thought process designing the OncoSeg architecture.**
> I started from the observation that brain tumors can be very large — spanning many slices. Pure CNNs have limited receptive fields, so I wanted a Transformer encoder for global context. But Transformers are parameter-heavy, so I paired it with a lightweight CNN decoder. The key innovation is the cross-attention skip connections — instead of just concatenating encoder and decoder features, the decoder learns to selectively attend to the most relevant encoder features.

**Q9: Why didn't you just use SwinUNETR off the shelf?**
> SwinUNETR uses simple concatenation for skip connections. I hypothesized that cross-attention would be more effective because it lets the decoder filter noise from the encoder features. To validate this, I included SwinUNETR as a baseline — same encoder, different skip connections. The ablation study is designed to isolate exactly this contribution.

**Q10: How did you implement the cross-attention skip connections?**
> I wrote a `CrossAttentionSkip` module. It flattens the 3D feature maps into sequences, applies LayerNorm, then computes multi-head attention where decoder features are queries and encoder features are keys/values. After the attention, there's a residual connection and an FFN. Then reshape back to 3D. I placed these at each decoder level except the lowest resolution, where a simple addition suffices.

**Q11: Why did you add Monte Carlo Dropout? Was that planned from the start?**
> It was planned from the start because in clinical settings, you need to know when the model is uncertain. I put a Dropout3d layer at the bottleneck. During inference, you can run multiple forward passes with dropout enabled — the variance across predictions tells you where the model is uncertain. Clinically, this means a radiologist can focus their review on uncertain regions.

**Q12: How did you decide on deep supervision?**
> Deep supervision helps with gradient flow in deep networks and forces intermediate features to be semantically meaningful. I added auxiliary loss heads at each decoder resolution. The weights decay exponentially — deeper outputs get lower weight. It was straightforward to implement and is standard in medical segmentation (nnU-Net uses it too).

**Q13: Why did you choose those specific baseline models?**
> Each baseline tests a specific hypothesis. UNet3D is pure CNN — tests if attention mechanisms add value at all. UNETR uses a ViT encoder — tests pure Transformer vs hybrid. SwinUNETR has the same Swin encoder as OncoSeg but standard skips — isolates the contribution of cross-attention. Together, they form a controlled experiment.

---

## The Softmax-to-Sigmoid Refactor

**Q14: Tell me about the biggest refactor you did and why.**
> I originally used softmax with 4 classes (background + 3 tumor regions). But BraTS tumor regions overlap — a voxel can be part of both TC and WT simultaneously. Softmax forces mutually exclusive predictions, which is fundamentally wrong here. I refactored to sigmoid multi-label: 3 channels, each independently predicting whether a voxel belongs to that region. This touched 15 files — configs, all models, losses, trainer, evaluator, inference.

**Q15: How did you catch that softmax was wrong?**
> I was reviewing the label conversion code and realized that WT contains TC which contains ET. If a voxel is ET, it's also TC and WT. Softmax can't express that — it would force the model to pick one class. This is a well-known issue in the BraTS community; the standard approach is sigmoid multi-label.

**Q16: What broke during the refactor?**
> The metrics. I had `include_background=False` in DiceMetric, which was correct when channel 0 was background. But after switching to 3 foreground-only channels (TC, WT, ET), it dropped channel 0 (TC) and left only 2 values. When the code tried to access `scores[2]`, it crashed with IndexError. Training ran for hours before hitting validation and crashing. I fixed it by setting `include_background=True` across all metric computations.

**Q17: What did you learn from that bug?**
> Always test the full pipeline — train AND validate — before starting long training runs. The bug only appeared during validation, not training. Also, when you do a sweeping refactor, trace every downstream assumption. "include_background=False" made sense in the old formulation but was wrong in the new one.

---

## Training Pipeline

**Q18: How did you build the training loop?**
> I wrote a Trainer class that handles the full lifecycle: loads model and data from Hydra config, sets up AdamW optimizer with cosine annealing scheduler, runs the training loop with gradient clipping, validates periodically with sliding window inference, logs to W&B, and saves checkpoints. It supports CUDA, MPS, and CPU.

**Q19: Why did you build train_all.py separately from the Trainer class?**
> The Trainer class is tightly coupled to Hydra configs — great for production but heavy for quick experiments. train_all.py is self-contained: it includes its own model definitions, doesn't require Hydra, and trains all 3 models sequentially with one command. I designed it for local M1 training where simplicity matters.

**Q20: Tell me about the checkpoint/resume system you built.**
> Training on M1 takes days — 50 epochs × 388 samples × 3 models. If the machine sleeps or the process crashes, you'd lose everything. So I added per-epoch checkpointing: saves model weights, optimizer state, scheduler state, and full training history. On restart, it detects checkpoints and resumes from the last completed epoch. Models that finished completely are skipped. The checkpoint file is deleted after successful completion to keep things clean.

**Q21: Why save optimizer and scheduler state, not just model weights?**
> If you only save model weights, resuming resets the learning rate back to the initial value and wipes AdamW's running averages (momentum). The model would essentially start over from a weird point — pretrained weights but freshman optimizer. Saving full state means the resume is seamless.

---

## Evaluation & RECIST

**Q22: How did you design the evaluation pipeline?**
> I built a SegmentationMetrics class that computes Dice, HD95, ASD, Sensitivity, and Specificity per region. The Evaluator class runs sliding window inference on the test set and feeds predictions to the metrics. I also built a result analyzer for comparing models and a failure analyzer to understand where and why the model fails.

**Q23: How did you approach building the RECIST assessment without clinical data?**
> I implemented the RECIST 1.1 algorithm on predicted segmentation masks — find connected components, measure longest axial diameter, compute volume, then classify response (CR/PR/SD/PD) by comparing baseline and follow-up measurements. Since I didn't have paired clinical data, I tested with synthetic geometric cases — empty masks, single voxels, spheres with known volume, cubes. The tests verify the algorithm is correct; clinical validation would come later.

**Q24: What's the limitation of your RECIST implementation?**
> It depends entirely on segmentation quality — if the segmentation is wrong, the measurement is wrong. It also doesn't handle non-measurable disease or complex multi-lesion scenarios fully. In a real clinical deployment, a radiologist would review the automated measurements. The uncertainty estimation from MC Dropout helps flag cases that need review.

---

## Testing & Quality

**Q25: How did you approach testing a deep learning project?**
> I test properties, not exact values. For models: check output shapes and that gradients flow. For losses: verify output is scalar, positive, and low for perfect predictions. For RECIST: use geometric shapes with known analytical solutions. I have 46 tests covering models, modules, losses, analysis, and response assessment. CI runs them on every push.

**Q26: Walk me through a test you're proud of.**
> The RECIST sphere test. I create a binary mask with a sphere of known radius, run RECIST measurement, and verify the computed volume matches 4/3πr³ within tolerance. It tests the full pipeline: connected component detection, volume calculation with anisotropic spacing. Simple setup, catches real bugs.

**Q27: How do you maintain code quality?**
> Ruff for linting and formatting — replaces flake8, black, and isort in one tool. Configured in pyproject.toml so everyone uses the same rules. CI enforces it on every push. The test suite catches regressions. Type hints throughout for clarity.

---

## Colab Notebook

**Q28: Why did you build a single self-contained notebook?**
> Reproducibility and accessibility. Anyone can open it in Google Colab, get a free T4 GPU, and run the entire pipeline — data download, training, evaluation, ablation, RECIST demo, visualization — without installing anything. It's 44 cells that tell the complete story.

**Q29: What went wrong with the notebook and how did you fix it?**
> I ran a macOS `sed -i ''` command to do a find-and-replace in the notebook. It wiped the file to zero bytes. Worse, I committed the empty file before noticing. I recovered it from git history, then applied the fix properly using Python JSON parsing instead of sed. Lesson: never use sed on structured files like JSON.

---

## Infrastructure & DevOps

**Q30: How did you set up CI/CD?**
> GitHub Actions workflow that runs on every push to main. Tests on Python 3.11 and 3.12 to catch version-specific issues. Steps: install dependencies, ruff lint check, ruff format check, run pytest. It caught a setuptools build backend issue early on.

**Q31: How did you add Apple Silicon support?**
> Added device auto-detection: CUDA → MPS → CPU. Handled MPS-specific issues like constant padding warnings in 3D and disabled multiprocessing data loading (num_workers=0). It's not fast enough for production training but invaluable for local development and testing.

---

## Debugging Stories

**Q32: Tell me about the hardest bug you debugged.**
> The include_background bug. Training ran for hours, then crashed during validation with a cryptic IndexError. The stack trace pointed to `scores[2]` being out of bounds for a size-2 tensor. I traced it back: DiceMetric with `include_background=False` was dropping channel 0, but after the sigmoid refactor, channel 0 was TC (foreground), not background. Changed to `include_background=True` across 5 files.

**Q33: Tell me about a time you broke something and had to recover.**
> The notebook corruption. A sed command emptied the 64KB Colab notebook. I'd already committed the empty file. Recovery: `git show <old_commit>:path > file` to restore from history. Applied fixes with proper JSON parsing. Added it as a lesson learned. Now I always use language-appropriate tools for structured file edits.

**Q34: What was the most impactful one-line change you made?**
> Changing `softmax=True` to `sigmoid=True` in the Dice Loss config. That single parameter change (along with the rest of the refactor) fixed the fundamental formulation error. The model went from trying to predict mutually exclusive classes to correctly predicting overlapping tumor regions.

---

## Decision-Making

**Q35: How did you decide what to build yourself vs use off-the-shelf?**
> I used MONAI for validated components: SwinTransformer, UNet, standard losses, metrics, transforms. I built custom: cross-attention skip connections (novel contribution), the RECIST pipeline (specific to our use case), the training orchestration (needed checkpoint/resume), and the analysis toolkit. The rule: build what's novel, reuse what's standard.

**Q36: How did you decide between training locally vs Colab?**
> I built both paths. Colab gives a free T4 GPU — fast enough for publication-grade results in 4-6 hours. Local M1 training is much slower (days) but useful for iterating on code without internet dependency. The checkpoint system made local training feasible despite the long runtime.

**Q37: How did you prioritize what to build?**
> I followed the dependency chain: structure → data → model → training → evaluation → RECIST → tests → notebook. Each step needed the previous one. Within each step, I built the minimum viable version first, then iterated. Testing came after the core pipeline worked, not after every component.

---

## Collaboration & Communication

**Q38: How would you explain this project to a non-technical person?**
> "We built AI that reads brain MRI scans and automatically outlines the tumor. It measures the tumor's size and can track whether it's growing or shrinking during treatment. It also tells doctors how confident it is, so they know when to double-check the AI's work."

**Q39: If a teammate joined this project tomorrow, how would they get up to speed?**
> Read the README for the high-level overview. Run the tests to verify the environment. Read the Pipeline Document for detailed architecture. Open the Colab notebook to see the full workflow. The Hydra configs show how components connect. The test files show how each component is used.

**Q40: What would you do differently next time?**
> Start with sigmoid from day one — the softmax-to-sigmoid refactor touched 15 files. Add learning rate warmup. Design for multi-GPU from the start. Write integration tests that run train + validate on 2 samples before any long training run. And never use sed on JSON files.
