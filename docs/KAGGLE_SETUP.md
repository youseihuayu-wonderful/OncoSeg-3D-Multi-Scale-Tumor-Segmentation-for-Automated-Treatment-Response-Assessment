# Running OncoSeg on Kaggle (free T4)

Kaggle is the recommended free host for finishing the remaining GPU work
(SwinUNETR + UNETR benchmarks and the 4-variant ablation). The free T4
session is more stable than free Colab and gives a full 9-hour window,
which fits the expected ~6-hour total wall time.

## What you need

- A Kaggle account (free): <https://www.kaggle.com/account/login>
- Phone verification on the account (required to enable GPU and Internet)
- The notebook `notebooks/OncoSeg_Full_Pipeline.ipynb` from this repo

## Step-by-step

### 1. Create a new Kaggle notebook

1. Go to <https://www.kaggle.com/code> → **New Notebook**.
2. Delete the default cell — you will upload your own notebook in step 3.

### 2. Enable GPU + Internet

In the right-hand **Session options** panel:

- **Accelerator** → `GPU T4 x2` (only one GPU is used, but the `x2` quota
  is the same queue and usually has less wait time).
- **Internet** → `On` (required to download MONAI + the MSD dataset).
- **Persistence** → `Files only` (keeps `/kaggle/working` between runs so
  you do not re-download the 7 GB dataset if the session restarts).

### 3. Upload the notebook

- Click the **File** menu → **Import Notebook** → **File** tab.
- Choose `notebooks/OncoSeg_Full_Pipeline.ipynb` from your local clone.
- Kaggle opens it. The first cell should be the markdown title.

### 4. Run all cells

- **Run All** from the **Run** menu.
- The notebook auto-detects Kaggle and uses `/kaggle/working` for outputs
  and `/kaggle/working/data` for the MSD dataset.
- Expected wall time on T4:
  - Dataset download + verification: ~15 min
  - OncoSeg training (50 epochs): already trained locally; re-runs if no
    checkpoint found (~2 h). Skip by uploading `oncoseg_best.pth` as a
    Kaggle dataset and pointing `CKPT_DIR` at it.
  - UNet3D + UNETR + SwinUNETR: ~3–4 h combined
  - 4-variant ablation (no_xattn / no_ds / no_mcdrop / small): ~1.5 h
  - Evaluation + figures: ~15 min
  - **Total: ~6 h**, well under the 9 h session cap.

### 5. Save the outputs

When the run finishes, everything you need is under `/kaggle/working`:

```
/kaggle/working/
├── checkpoints/
│   ├── oncoseg_best.pth
│   ├── unet3d_best.pth
│   ├── unetr_best.pth
│   └── swinunetr_best.pth
├── results.csv
├── ablation_results.csv
├── evaluation_results.json
├── experiment_config.json
├── training_curves.png
├── ablation_comparison.png
├── recist_longitudinal_demo.png
└── *_history.json   (one per model)
```

Two ways to get the files back to your laptop:

**A. Commit + version (recommended)**
1. Click **Save Version** (top-right) → **Save & Run All (Commit)**.
   Kaggle re-runs the notebook one more time end-to-end in the
   background and snapshots every output file into a version.
2. When the version finishes, open it and use the **Output** tab →
   **Download all** (zip).

**B. Download individual files**
- Right-hand **Output** panel in the editor → click any file → download.
- Best for grabbing just the `.pth` checkpoints without re-running.

## If the session disconnects

- The checkpoint/resume logic in `train_all.py` (and mirrored in the
  notebook) skips any model whose `*_best.pth` already exists in
  `CKPT_DIR`. With **Persistence = Files only** enabled, restarting the
  kernel picks up exactly where it left off.
- If Kaggle kills the kernel mid-epoch, you lose that epoch but not the
  previous best — re-run the notebook, it will resume at the next model.

## What to send back

Upload these to the repo (or share via Drive / a Kaggle dataset) so the
paper's Results tables can be filled in:

1. `/kaggle/working/checkpoints/*.pth` — all four main checkpoints plus
   the four ablation checkpoints (8 files).
2. `/kaggle/working/results.csv` + `ablation_results.csv`.
3. `/kaggle/working/evaluation_results.json` — per-model Dice / HD95 /
   parameter counts.
4. `/kaggle/working/*.png` — training curves, ablation comparison,
   RECIST demo.

Drop them in `experiments/kaggle_run/` (create the directory) and we
will wire the numbers into the README + paper.

## Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `!pip install` fails, `Network is unreachable` | Internet not enabled | Session options → Internet → On |
| `No module named 'monai'` | pip cell skipped | Re-run the install cell, restart kernel |
| `CUDA out of memory` on UNETR | ROI too large for T4 (16 GB) | Reduce `roi_size` to `(96, 96, 96)` in the transform cell |
| Session hits 9 h cap mid-ablation | Training slower than estimated | Restart kernel — checkpoint/resume skips finished models |
| Dataset download aborts | S3 hiccup | Re-run the download cell; `wget` will resume from a partial tar if you `rm` the dir first |

## Why not Colab this time?

Free Colab reclaims idle sessions and drops connections after ~90 min of
inactivity during training; the 50-epoch OncoSeg run on 2026-04-06 did
not complete for that reason. Kaggle's 9-hour policy is more forgiving,
and the persistent `/kaggle/working` directory survives kernel
restarts, so a partial run is recoverable instead of lost.
