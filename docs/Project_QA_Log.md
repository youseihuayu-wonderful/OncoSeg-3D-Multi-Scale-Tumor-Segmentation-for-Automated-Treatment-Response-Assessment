# OncoSeg Project — Questions & Answers Log

A record of all questions asked by both the user and Claude throughout the project.

> **Note:** Questions from earlier sessions (Steps 1–28) are reconstructed from context and memory. Questions from the current session (2026-04-04/05) are recorded verbatim.

---

## Phase 1: Project Setup & Architecture (Steps 1–4)

**User:** How should we structure the OncoSeg project?
> Claude set up the repo with pyproject.toml, Hydra configs, src/ package layout, and all model configs.

**User:** What architecture should we use?
> Hybrid Swin Transformer encoder + CNN decoder with cross-attention skip connections, MC Dropout uncertainty, and deep supervision.

**User:** Should we include baseline models?
> Yes — UNet3D, UNETR, SwinUNETR wrappers from MONAI for comparison.

---

## Phase 2: Data & Training Pipeline (Steps 5–9)

**User:** What datasets should we support?
> MSD Brain Tumor (primary), BraTS 2023, KiTS23, LiTS, BTCV.

**User:** How should we handle MSD's 4D NIfTI format?
> Custom transforms to split 4-channel images and convert labels to multi-channel (TC, WT, ET).

**User:** What evaluation metrics do we need?
> Dice, HD95, ASD, Sensitivity, Specificity — per region (ET, TC, WT).

---

## Phase 3: Treatment Response & Inference (Steps 7, 10)

**User:** How do we automate RECIST measurement?
> Implemented longest axial diameter calculation, volume measurement, and CR/PR/SD/PD classification from baseline vs follow-up masks.

---

## Phase 4: Colab Notebook (Steps 11, 22)

**User:** Can we make everything run in a single Colab notebook?
> Yes — 44-cell self-contained notebook that downloads MSD data, trains all models, evaluates, runs ablation, RECIST, visualization, and statistical tests.

---

## Phase 5: Bug Fixes & Quality (Steps 13–16, 23–26)

**User:** Are there bugs in the model architecture?
> Yes — fixed SwinEncoder3D, CNNDecoder3D, DeepSupervisionHead, SwinUNETR baseline (Step 13).

**User:** Do all tests pass?
> Expanded from 7 to 46 tests, all passing. Ruff lint clean, CI green on Python 3.11 + 3.12.

**User:** Can we remove unused dependencies?
> Removed SimpleITK, pydicom, transformers, tokenizers, fastapi, etc. Added missing einops.

---

## Phase 6: Local Training & MPS Support (Steps 21–22)

**User:** Can we train locally on Apple Silicon?
> Added MPS GPU support to trainer, evaluator, and inference. Created train_local.py with M1-safe settings.

---

## Phase 7: Documentation (Steps 12, 20, 27–28)

**User:** Can you write an AI Knowledge Fundamentals document?
> Created 1500+ line comprehensive reference at docs/AI_Knowledge_Fundamentals.md.

**User:** Can you write a paper methods section?
> Created 200-line draft at docs/Paper_Methods_Draft.md with references.

---

## Current Session (2026-04-04 / 2026-04-05)

### User Questions

**Q: "counite" (continue)**
> Claude reviewed project state, reported uncommitted changes (sigmoid refactor + train_all.py), and listed next steps.

**Q: "could you please do all of it" (run tests, commit, push)**
> Ran 46 tests (all passing), committed sigmoid refactor + train_all.py, pushed to main.

**Q: "can you git and pull"**
> Ran git pull — already up to date.

**Q: "okay, what should we do next step"**
> Listed 4 remaining tasks: (1) train models, (2) fill results, (3) generate figures, (4) iterate. Also listed 4 things we could do while waiting on training.

**Q: "yes please" (start local training)**
> Started train_all.py on MSD data (50 epochs, 3 models, M1 MPS).

**Q: "is the training are very important?"**
> Yes — without it, there are no weights, no metrics, no figures. The project is just code with no proof it works. Recommended Colab for speed.

**Q: "if you end of the training, then did the project will not be continue to?"**
> No — killing local training doesn't stop the project. The code stays. Just need to run training on a faster GPU (Colab T4).

**Q: "do not kill the local run, just keep to do training"**
> Kept training running in background on M1.

**Q: "what you could do for next?"**
> Proposed 4 tasks: (1) results templates, (2) verify Colab notebook, (3) integration tests, (4) checkpoint/resume support. Recommended #4 as most valuable.

**Q: "yes, start to do that" (resume support)**
> Added checkpoint/resume to train_all.py — saves every epoch, resumes on restart, skips completed models.

**Q: "are you doing something to closer to the goal?"**
> Honestly no — was just waiting on training. Checked progress, found it was still on epoch 1 after 1.5 hours.

**Q: "before we start to do things, tell me how this project going so far"**
> Gave full status report: 28 steps completed, all code built, 46 tests passing, but no trained weights or real results yet. Training running for first time.

**Q: "please create a file, and put all questions that we both ask in this whole project"**
> Created this file.

### Claude Questions

**Q: "Did you mean continue?"** (after user typed "counite")
> User confirmed by asking to proceed.

**Q: "Would you like me to: Commit and push? Run tests first? Something else?"**
> User said "could you please do all of it."

**Q: "Would you like to start training locally?"**
> User said "yes please."

**Q: "Want me to stop the current run and restart with a fast smoke-test config?"**
> User said no — keep training, don't kill it.

**Q: "Want me to kill the local training? Then you can upload the notebook to Colab."**
> User said don't kill it.

**Q: "Is there anything else you'd like me to work on in the meantime?"**
> User asked what could be done next.

**Q: "Want me to start with that?" (checkpoint/resume support)**
> User said "yes, start to do that."

**Q: "Want me to keep working on other things, or wait for training to finish?"**
> User asked to create this Q&A log file.

---

## Key Decisions Made

| Decision | Rationale |
|----------|-----------|
| Sigmoid (multi-label) over softmax (multi-class) | BraTS tumor regions overlap (TC, WT, ET) — sigmoid is the correct formulation |
| num_classes: 4 → 3 | Dropped background channel; predict TC/WT/ET directly |
| include_background=True for metrics | All 3 channels are foreground in multi-label mode |
| Keep local M1 training running | User prefers patience over switching to Colab |
| Add checkpoint/resume | Long training on M1 needs crash protection |
| Deep learning only, no classical ML | Project scope decision — research-grade medical AI |

---

## Open Questions (Unresolved)

1. Will OncoSeg outperform baselines on real data? (Training in progress, best Dice so far: 0.7387)
2. Should we re-run on Colab with full settings (embed_dim=48) for publication-grade results?
3. Will temporal attention improve longitudinal scan comparison?
4. Is there clinical data available for RECIST validation?
