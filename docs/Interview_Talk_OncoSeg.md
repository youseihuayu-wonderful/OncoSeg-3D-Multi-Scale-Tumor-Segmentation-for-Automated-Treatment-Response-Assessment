# OncoSeg — 2–3 Minute Interview Talk (STAR)

Spoken script (English) + delivery notes. ~2.5 minutes at a normal speaking pace.
All numbers are real and traceable to `README.md` Results / `experiments/local_results/`.

---

## ① PROBLEM (~30s)
> "In oncology, the way we decide whether a treatment is working is by measuring the patient's
> tumor on CT or MRI scans over time. Today that's done **manually** — a radiologist outlines the
> tumor by hand. It's **slow** (15–30 minutes per scan), **subjective** — two radiologists disagree
> by 20–40% on where the tumor edge is — and it's traditionally reduced to a single **2D** diameter,
> which throws away most of the 3D structure. In a clinical trial with thousands of scans, that's a
> real bottleneck and a source of noise in the trial's outcome."

## ② APPROACH (~50s)
> "I built **OncoSeg** to automate this end-to-end, around three design ideas:
> - First, **multi-scale 3D segmentation** — a Swin-Transformer encoder captures global context, a
>   CNN decoder recovers fine detail, and **cross-attention skip connections** fuse the two. It
>   segments the whole tumor in 3D, not a single slice.
> - Second, **uncertainty** — using Monte-Carlo Dropout, the model outputs not just a mask but an
>   **uncertainty map**, so it can flag the cases where a human should double-check. In a clinical
>   setting, knowing *when* the model might be wrong matters as much as the prediction itself.
> - Third, **temporal response assessment** — the pipeline takes the 3D segmentations across
>   timepoints and automatically computes the standard **RECIST 1.1** category: shrinking, stable,
>   or growing."

## ③ RESULTS (~40s) — show 2 figures
> "Two results I'm proud of. **One — efficiency:** on the Medical Segmentation Decathlon brain-tumor
> set, OncoSeg matches a standard 3D U-Net's accuracy (Dice 0.797) while using **5× fewer
> parameters** — 3.7 million versus 19 million — and gets **27% more accurate tumor boundaries**
> (HD95 15.4 mm vs 21.0 mm). So it's lighter and sharper at the same time.
> **Two — the uncertainty is calibrated:** expected calibration error is about **0.01**, and on the
> failure cases the model's uncertainty lines up with where it's actually wrong. That's what makes
> the uncertainty map trustworthy rather than decorative."

## ④ IMPACT (~25s)
> "The impact is two-fold: it can **accelerate clinical trials** by removing the manual-measurement
> bottleneck and making response assessment consistent across sites, and it **reduces radiologist
> burden** on a repetitive task — while keeping a human in the loop exactly where the model is
> uncertain. More broadly, my interest is in **making AI reliable enough to trust in high-stakes
> settings**, and medical imaging is where I started building that."

---

## Figures to show on screen
| When | Figure | Path |
|---|---|---|
| Results pt.1 | Dice + parameter comparison | `experiments/local_results/dice_comparison.png` |
| Results pt.2 | reliability diagram / uncertainty-vs-error | `experiments/local_results/` |

## Honesty guardrails (do NOT claim)
- Do **not** say "beats SwinUNETR/UNETR" — those benchmarks are not run yet; only UNet3D was the real head-to-head.
- Do **not** claim temporal validated on real longitudinal data — the LUMIERE real-data evaluation is built but not yet run; temporal = a demonstrated *capability* (RECIST demo), not a measured result.
- Say "matches U-Net" not "greatly outperforms" — Dice is only +0.0025; the real wins are 5× fewer params + 27% lower HD95 + calibrated uncertainty.

## Likely follow-ups + crisp answers
1. **"Why only beat UNet3D, not the transformer baselines?"**
   → "Those need a CUDA GPU; I trained locally on Apple Silicon, so UNet3D was my honest head-to-head. The SwinUNETR/UNETR benchmarks are wired up and ready on Colab/Kaggle — I just won't quote numbers I haven't measured."
2. **"How do you know the uncertainty is meaningful, not noise?"**
   → "I checked calibration — ECE ~0.01 — and plotted uncertainty against actual error on the failure cases; high-uncertainty regions are where the model is actually wrong. Calibrated, not just high-variance."
3. **"Biggest limitation?"**
   → "Single dataset, single tumor type so far, and the longitudinal response validation is on a ready-but-not-yet-run real cohort (LUMIERE). Honest next steps, not solved problems."
