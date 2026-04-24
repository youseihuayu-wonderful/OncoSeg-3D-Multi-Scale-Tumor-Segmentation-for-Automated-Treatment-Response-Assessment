# OncoSeg — Project Overview

**Last updated:** 2026-04-24
**Repo:** https://github.com/youseihuayu-wonderful/OncoSeg-3D-Multi-Scale-Tumor-Segmentation-for-Automated-Treatment-Response-Assessment
**Branch:** `main` (67 commits)
**Test suite:** 142 tests across 12 files
**Trained artefacts on disk:** `experiments/local_results/oncoseg_best.pth`, `unet3d_best.pth`

This file is a single-source map of everything that has been built, what works, what is still open, and where the risks are. It complements (does not replace) `README.md`, `docs/Pipeline_Document.md`, and the paper drafts.

---

## 1. What OncoSeg is

A 3D multi-scale tumor segmentation system for automated treatment-response assessment in oncology trials.

**Architecture.** Hybrid CNN-Transformer: Swin Transformer encoder + CNN decoder, cross-attention skip connections, Monte Carlo Dropout for uncertainty, deep supervision.

**Downstream.** Segmentation → per-lesion RECIST 1.1 measurements → response class (CR / PR / SD / PD).

**Scope.** Deep-learning only (no classical-ML baselines). Research-grade, not an introductory-ML survey.

**Stack.** Python 3.11+, PyTorch 2.1+, MONAI 1.3+, Hydra/OmegaConf, FastAPI, W&B, pytest, Ruff, mypy.

---

## 2. Repository layout

```
configs/                Hydra configs: config.yaml + data/ model/ experiment/
src/
├── data/               MSD, BraTS, LUMIERE, DICOM loaders + transforms
├── models/             oncoseg.py, baselines/ (UNet3D, UNETR, SwinUNETR), modules/
├── training/           trainer.py, losses.py
├── evaluation/         evaluator.py, metrics.py (Dice, HD95)
├── analysis/           failure_analyzer, figures, model_profiler, result_analyzer
├── response/           recist.py, classifier.py (CR/PR/SD/PD)
├── api/                FastAPI service: app, service, schemas, cli
└── inference.py        Predictor (sliding-window, MC Dropout)
scripts/                verify_msd_dataset, run_ablation, dryrun_ablation,
                        evaluate_lumiere, uncertainty_qualitative_analysis,
                        diagnose_worst_case, integrate_kaggle_results
notebooks/              OncoSeg_Full_Pipeline.ipynb (Kaggle/Colab/local auto-detect)
                        Ablation_Study.ipynb (untracked)
                        recist_response_demo.ipynb
tests/                  142 tests, 12 files
experiments/local_results/  Trained checkpoints + eval JSON + figures
figures/                Qualitative, uncertainty, RECIST figures
paper/                  Paper drafts (Methods, Results) + figures
docs/                   Pipeline, interview banks, Kaggle setup, paper drafts
train_all.py            Multi-model local trainer (checkpoint/resume)
train_local.py          Single-model local trainer
Dockerfile              FastAPI service container
```

---

## 3. Completed work — chronological

All line items reference the commit that landed them. Dates are the commit date.

### Phase A — Foundation (2026-03 → 2026-04-03)
Steps 1–28 across many commits (see `git log`). Delivered:
- Full package scaffold (`src/`, `configs/`, `tests/`, `scripts/`)
- MSD Brain Tumour loader + transforms, BraTS loader
- OncoSeg architecture + baselines (UNet3D, UNETR, SwinUNETR)
- Training pipeline (trainer + losses including Dice + focal)
- Evaluation (Dice, HD95, per-region)
- RECIST 1.1 measurement + response classifier
- Hydra configs, GitHub Actions CI, pre-commit, Ruff, pyproject
- Colab notebook, 46 tests, READMEs, pipeline doc

Key commits: `4f4b3db` (init), `fc050c3` (MSD loader), `f419141` (inference + notebook), `f7c6e96` (critical model fixes), `bdb8d39` (tests → 31), `f299c2b` (temporal + KiTS23/LiTS/BTCV configs), `d15bc6e` (analysis toolkit), `435cd05` (CI + analysis tests → 46), `82476e8` (profiler + pre-commit), `efe980e` (`Pipeline_Document.md`), `d788ba4` (MPS + MSD config + notebook rewrite).

### Step 29 — Multi-label sigmoid refactor (`f2d3d46`, 2026-04-04)
Switched loss/metrics from 4-class softmax to 3-channel sigmoid (TC / WT / ET). `include_background=True` in metrics. Added `train_all.py` for multi-model local training.

### Step 30 — Checkpoint / resume (`8486805`, 2026-04-05)
Per-epoch checkpoints in `train_all.py`; completed models skipped on restart.

### Step 31 — Colab notebook restore (`4a4b30b`, 2026-04-05)
Notebook corrupted by a prior `sed` edit — restored from git and re-applied the `include_background` fix via JSON parsing.

### Step 32 — OncoSeg training completed (2026-04-06)
50 epochs on MSD Brain Tumour (388 train / 96 val).
**Best mean Dice 0.7969 at epoch 50** — TC 0.7898, WT 0.8529, ET 0.7481.

### Step 33 — UNet3D baseline (2026-04-06)
~30 epochs before Colab OOM, best checkpoint saved.
TC 0.7849, WT 0.8522, ET 0.7462, Mean 0.7944.
**OncoSeg beats UNet3D on all 3 regions with 5× fewer params** — key comparison for paper.

### Step 34 — Training figures (`7061d20`, 2026-04-06)
`training_curves.png` (loss + Dice vs epoch), `dice_comparison.png` (per-region bar chart).

### Step 35 — Results in README + Paper Methods (`27029e6`, `75af64b`, `de43bc9`, 2026-04-06)
Real numbers embedded. Training-dynamics analysis written.

### Steps 36–37 — Interview prep + resume (2026-04-05/06)
`docs/Interview_Questions.md` (100), `Interview_AI_Fundamentals.md` (90), `Interview_LeetCode_Questions.md` (25), `Project_QA_Log.md` (40), `AI_ML_Hiring_Points.md`, `Manager_Interview_Questions.md` (72), `~/Desktop/Resume_AI_ML_ShihuaYu.md`. Not scientific content — career artefacts built alongside the project.

### Steps 38–41 — Evaluation story + RECIST demo (`59979d4`, `5140115`, `de4a1a1`, `d7504f5`, 2026-04-13)
- `scripts/uncertainty_qualitative_analysis.py`: best/median/worst qualitative comparison (BRATS_407 / 425 / 077), MC Dropout on median case (**ECE = 0.0101**), reliability diagram, uncertainty-vs-error plot, bottom-5 failure analysis (dominant failure = TC, 79.7% relative drop).
- `notebooks/recist_response_demo.ipynb`: end-to-end RECIST demo, simulated PR/SD/PD all classify correctly.
- HD95 + Wilcoxon significance tests (`de4a1a1`), ablation dry-run harness (`d7504f5`).

### Step 42 — Ablation + realistic longitudinal RECIST (`330142d`, 2026-04-17)
Colab notebook now trains 4 ablation variants via unified `build_model` factory: `oncoseg`, `no_xattn`, `no_ds`, `no_mcdrop`, `small`. `scripts/run_ablation.py` for standalone GPU. RECIST demo uses biologically-motivated evolution: exponential peripheral decay, Gompertz growth, heterogeneous subclonal response across 4 cycles.

### Step 43 — LUMIERE loader + RECIST inference fix (`561b804`, 2026-04-18)
`src/data/lumiere.py`: 91 GBM patients, 638 timepoints, 4 modalities + RANO CSV. Tolerates missing modalities and CSV column-name variants. `src/inference.py`: fixed RECIST ET extraction (legacy softmax label 3 → sigmoid channel [2]). `tests/test_lumiere.py`: 14 tests.

### Step 44 — LUMIERE longitudinal evaluator (`c5d2b7f`, 2026-04-18)
`scripts/evaluate_lumiere.py`: loads checkpoint → runs every timepoint → classifies CR/PR/SD/PD vs baseline → vs expert RANO → emits `per_visit.csv` + `summary.json` (accuracy, Cohen κ) + `confusion_matrix.png`. `tests/test_evaluate_lumiere.py`: 22 tests incl. end-to-end orchestration with deterministic in-process model.

### Step 45 — README: LUMIERE section + Colab benchmark table (`32fc68f`, 2026-04-18)
Usage + Figshare pointer; T4 wall-time estimates for SwinUNETR / UNETR / 4-variant ablation.

### Step 46 — Kaggle port + UNETR factory (`fd3313f`, `9a596d0`, 2026-04-18)
`train_all.py build_model` registers MONAI UNETR (hidden_size=768, mlp_dim=3072, num_heads=12, img_size pinned to ROI). `notebooks/OncoSeg_Full_Pipeline.ipynb` auto-detects Kaggle / Colab / local via `/kaggle/working` vs `/content` vs `cwd`; all hardcoded `/content` paths rewritten to `OUT_ROOT` / `CKPT_DIR`. Same notebook runs unchanged on Colab. `docs/KAGGLE_SETUP.md`: click-by-click Kaggle guide (GPU + Internet toggles, 6 h budget, Save Version vs per-file download, checkpoint/resume on disconnect). Motivation: **2026-04-06 Colab run was killed by free-tier idle reclamation**.

### Step 47 — build_model regression test suite (`ebaad34`, 2026-04-18)
`tests/test_build_model.py`: 11 tests covering all 8 factory names (oncoseg, 4 ablation variants, unet3d, swin_unetr, unetr). Each instantiates + forward-passes small CPU tensor, asserts `[B, 3, *ROI]`. Semantic invariants: `oncoseg_small` strictly fewer params than baseline; `oncoseg_no_xattn` different param count (confirms xattn knob wired). Protects the 6-hour Kaggle run from crashing on a config bug after minutes of data download.

### Step 48 — CI fixes (`f2ba2c9`, `dccf37b`, reverted `5c23ed7` → `e5a3c12`, 2026-04-18)
CI on `main` was failing for 5+ consecutive pushes — pytest passed but `ruff check src/ tests/` failed with I001 unsorted-imports on 3 test files. Two rounds of fixes caught them all.
Briefly shipped `docs/Paper_Discussion_Draft.md` (`5c23ed7`) with `[Kaggle TBD]` placeholders + speculative discussion narrative. Flagged as mock content; reverted (`e5a3c12`). Added `feedback_no_paper_drafts_with_placeholders.md` to memory so future sessions don't repeat.

### Step 49 — MSD pre-flight checker (`36849a5`, 2026-04-18)
`scripts/verify_msd_dataset.py`: validates Task01_BrainTumour before a multi-hour GPU run. Checks dir structure, `dataset.json numTraining` vs list length, file count ≥ 380, per-sample NIfTI validity (loads, 4 modalities, spatial ≥ 32, image/label shape match, label values ⊆ {0,1,2,3}). Exit 0 pass, 1 fail. `tests/test_verify_msd_dataset.py`: 13 tests via `tmp_path` + synthetic NIfTI.

### Step 50 — FastAPI inference service + Docker (`3f54309`, 2026-04-22)
`src/api/`: `app.py` (FastAPI), `service.py` (wrapping Predictor + RECIST + classifier), `schemas.py` (pydantic), `cli.py` (`oncoseg-serve` entry).
Endpoints: `GET /healthz /readyz /info`; `POST /predict/segment` (NIfTI out), `/predict/measure` (channel stats + RECIST JSON), `/predict/response` (CR/PR/SD/PD).
Dual model-source: `--model-source train_all` (matches `local_results/oncoseg_best.pth`) or `src` (matches `src.models.oncoseg`). Deferred imports so train_all deps only load when serving its checkpoints.
`Dockerfile` (python:3.11-slim, non-root, healthcheck), `.dockerignore`; checkpoint mounted via `ONCOSEG_CHECKPOINT` env var.
pyproject `[serve]` extra (fastapi, uvicorn[standard], python-multipart) + `oncoseg-serve` console script.
`tests/test_api.py`: 12 tests via `TestClient` + `FakePredictor` (no GPU/checkpoint needed). Covers health/ready/info (+ 503 when unloaded), measure happy path (RECIST lesion counts on deterministic 2-lesion seg), segment NIfTI round-trip, response SD (identical) + CR (empty followup), missing modality (422), empty upload (400), bad model source (ValueError).

### Kaggle training iteration — v3 through v9 (2026-04-18)
Notebook ran on Kaggle and hit several failures, each fixed:
- `0e64fc4` — v1: Kaggle-compat GPU cells (sm_60 guard, safe device-props access)
- `d2013d4` — v3: reinstall torch via nvidia-smi-first guard
- `42fb176` — v4: clean-uninstall `torch*` before reinstall on P100
- `f211679` — pin Kaggle GPU to T4 via `machine_shape`, revert setup cells
- `6dd2806` — v6: notebook OncoSeg used wrong `stage_features` slice
- `dd70088` — v7: train SwinUNETR + UNETR only, persist artifacts mid-run
- `c138ccd` — v7 fix: training cell missing `MAX_EPOCHS` + `VAL_INTERVAL` decls
- `1b9a384` — v8: OOM → batch 2→1 and `SwinUNETR(use_checkpoint=True)`
- `51c6621` — v9 artefacts: `scripts/integrate_kaggle_results.py` wires outputs back into repo

### Other (interview / meta)
- `675b0a5` — 72-question manager-angle interview bank (2026-04-18)
- `d711d2d`, `7f65e36`, `750df9b`, `6ccf60b`, `fae6d5b`, `8ea2648` — interview Q&A banks
- `6f85b81`, `7631d45` — ruff cleanups post-refactor

---

## 4. Current state

### Working tree (dirty)
```
modified:   README.md
modified:   pyproject.toml
modified:   src/api/app.py
modified:   src/api/cli.py
modified:   src/api/service.py
modified:   tests/test_api.py
Untracked:  notebooks/Ablation_Study.ipynb
            src/data/dicom.py
            tests/test_dicom.py
```
These are post-`3f54309` API polish + a new DICOM loader + new Ablation notebook, not yet committed. Needs review before the next push.

### Trained artefacts (on disk, under `experiments/local_results/`)
- `oncoseg_best.pth` — 50 epochs, mean Dice **0.7969**, HD95 mean 15.35
- `unet3d_best.pth` — mean Dice **0.7944**, HD95 mean 21.03
- `oncoseg_eval.json`, `unet3d_eval.json`, per-subject Dice `.npy`
- `training_curves.png`, `dice_comparison.png`
- `uncertainty_metrics.json` (ECE 0.0101), `failure_analysis.json`, `worst_case_diagnosis.json`

### Tests
142 tests across 12 files: `test_analysis`, `test_api`, `test_build_model`, `test_dicom`, `test_evaluate_lumiere`, `test_integrate_kaggle_results`, `test_losses`, `test_lumiere`, `test_models`, `test_modules`, `test_response`, `test_verify_msd_dataset`.

---

## 5. What is NOT done

1. **SwinUNETR benchmark on MSD** — needs Kaggle T4 run via `notebooks/OncoSeg_Full_Pipeline.ipynb`.
2. **UNETR benchmark on MSD** — same notebook, factory already registered.
3. **Ablation study full training** — 4 variants (`no_xattn`, `no_ds`, `no_mcdrop`, `small`) inside the same notebook on Kaggle.
4. **LUMIERE evaluation execution** — loader + evaluator ready; need to download from Figshare and run `python scripts/evaluate_lumiere.py --lumiere-root <path> --checkpoint experiments/local_results/oncoseg_best.pth`.
5. **Paper Discussion + Conclusion** — deliberately not drafted until 1–4 produce real numbers (memory: never write paper text ahead of results).

---

## 6. Known risks and gotchas

1. **Trained checkpoints were produced by `train_all.py`'s inline OncoSeg class — NOT `src/models/oncoseg.py`.** The two architectures diverge (MONAI `SwinTransformer` vs custom `SwinEncoder3D`, different decoder shape). Any script loading the checkpoints must `from train_all import OncoSeg`, not the src module. Wasted ~2 iterations on this. FastAPI service handles both via `--model-source train_all|src`.
2. **Val subjects with NaN Dice (empty GT regions) and all-zero Dice (complete failure)** — must be filtered before best/median/worst case selection.
3. **Colab free-tier idle reclamation** — killed the 2026-04-06 run and the 2026-04-24 run (~46 min before kernel died, see `~/Downloads/app.log`). Plan is **Kaggle T4 only** for training now (`docs/KAGGLE_SETUP.md`).
4. **No AMP in the notebook training loop** — no `autocast` / `GradScaler`. 128³ ROI on SwinUNETR/UNETR is borderline on T4 16 GB without mixed precision. Adding AMP is the biggest remaining memory/throughput win.
5. **Training-stdout capture gap** — `app.log` is Jupyter server log, not cell output. Future runs need `tee` or `FileHandler` so stdout persists through a kernel death.
6. **Paper drafts in `docs/` must not contain `[TBD]` placeholders or speculative narrative** — see `5c23ed7` → `e5a3c12` revert. Draft only after real numbers land.

---

## 7. Next session — recommended actions

Priority order:

1. **Review and commit** the 6 dirty files + 3 untracked (`notebooks/Ablation_Study.ipynb`, `src/data/dicom.py`, `tests/test_dicom.py`) so `main` is clean before the next Kaggle push.
2. **Add AMP** (`autocast` + `GradScaler`) to `notebooks/OncoSeg_Full_Pipeline.ipynb` training cell + a file-based stdout logger. Commit.
3. **Launch Kaggle run** per `docs/KAGGLE_SETUP.md` for SwinUNETR + UNETR + 4 ablation variants.
4. **Integrate results** via `scripts/integrate_kaggle_results.py`; run `scripts/uncertainty_qualitative_analysis.py` + `diagnose_worst_case.py` as needed.
5. **LUMIERE** — download from Figshare, run `evaluate_lumiere.py`, get per-visit CSV + confusion matrix + κ.
6. **Paper Discussion + Conclusion** once 3–5 produce real numbers.
