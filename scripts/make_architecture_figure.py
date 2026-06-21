"""Generate formal architecture diagram + infographic for OncoSeg.

Renders two publication-quality figures into figures/:
  - architecture_diagram.{png,svg}: U-shaped hybrid Swin-encoder / CNN-decoder net
  - infographic.{png,svg}: one-page design-rationale infographic

Pure matplotlib (no extra deps). Numbers mirror src/models/oncoseg.py defaults
and the trained results recorded in the progress tracker.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.font_manager as fm  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

plt.rcParams["font.family"] = "DejaVu Sans"

# ---- palette -----------------------------------------------------------------
C_SWIN = "#3D5A9E"      # encoder (Swin Transformer)  - indigo
C_SWIN_L = "#9FB1D8"
C_CNN = "#2E8B72"       # decoder (CNN)               - teal/green
C_CNN_L = "#A6D4C6"
C_XATTN = "#E08A2B"     # cross-attention skip        - amber
C_BOTTLE = "#7B4FA3"    # bottleneck / temporal       - purple
C_HEAD = "#C0436B"      # output / RECIST head        - rose
C_INK = "#1c2430"
C_BG = "#FFFFFF"
C_PANEL = "#F4F6FA"

SHADOW = [pe.withSimplePatchShadow(offset=(2, -2), alpha=0.18)]


def rbox(ax, x, y, w, h, fc, ec=None, lw=1.4, r=0.035, z=2, shadow=True):
    ec = ec or fc
    p = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={r}",
        linewidth=lw, edgecolor=ec, facecolor=fc, zorder=z,
        path_effects=SHADOW if shadow else None,
    )
    ax.add_patch(p)
    return p


def txt(ax, x, y, s, size=11, color=C_INK, weight="normal", ha="center",
        va="center", style="normal", z=5):
    ax.text(x, y, s, fontsize=size, color=color, fontweight=weight, ha=ha,
            va=va, style=style, zorder=z)


def arrow(ax, p0, p1, color=C_INK, lw=2.0, style="-|>", rad=0.0, z=3, ls="-"):
    a = FancyArrowPatch(
        p0, p1, arrowstyle=style, mutation_scale=16, lw=lw, color=color,
        connectionstyle=f"arc3,rad={rad}", zorder=z, linestyle=ls,
        shrinkA=2, shrinkB=2,
    )
    ax.add_patch(a)
    return a


# ============================================================================
# FIGURE 1 — U-shaped architecture
# ============================================================================
def architecture():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=150)
    fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    # title
    txt(ax, 0.35, 9.05, "OncoSeg", size=27, weight="bold", color=C_INK, ha="left")
    txt(ax, 2.95, 9.08, "Hybrid Swin-Transformer Encoder  /  CNN Decoder  with Cross-Attention Skips",
        size=13.5, color="#55617a", ha="left", weight="bold")
    txt(ax, 0.37, 8.62, "3D multi-scale tumor segmentation  •  input  [B, 4, H, W, D]  (T1, T1c, T2, FLAIR)   →   output  [B, 3, H, W, D]  (TC / WT / ET, sigmoid)",
        size=10.5, color="#7a8499", ha="left")

    # geometry: U-shape. encoder columns descend left->bottom, decoder ascend right.
    # stage rows (y centers) for the 4 scales + bottleneck row
    rows = {
        "s1": 7.05,   # H/4   48
        "s2": 5.85,   # H/8   96
        "s3": 4.65,   # H/16  192
        "s4": 3.45,   # bottleneck H/32 384
    }
    enc_x = 2.5
    dec_x = 11.6
    bw, bh = 2.0, 0.78

    enc_info = [
        ("Swin Stage 1", "48 ch · H/4", "depth 2 · 3 heads"),
        ("Swin Stage 2", "96 ch · H/8", "depth 2 · 6 heads"),
        ("Swin Stage 3", "192 ch · H/16", "depth 6 · 12 heads"),
        ("Swin Stage 4", "384 ch · H/32", "depth 2 · 24 heads"),
    ]
    dec_info = [
        ("Decoder Block 1", "ConvT3d ↑2 · 192 ch", None),
        ("Decoder Block 2", "ConvT3d ↑2 · 96 ch", None),
        ("Decoder Block 3", "ConvT3d ↑2 · 48 ch", None),
    ]

    # ---- patch embed (input stem) -------------------------------------------
    rbox(ax, 0.45, rows["s1"] - 0.05, 1.55, 0.9, "#E9EEF7", ec=C_SWIN, lw=1.6)
    txt(ax, 1.22, rows["s1"] + 0.58, "Patch Embed", size=10, weight="bold", color=C_SWIN)
    txt(ax, 1.22, rows["s1"] + 0.27, "4×4×4 patch", size=8.5, color="#55617a")
    txt(ax, 1.22, rows["s1"] + 0.02, "↓4×", size=8.5, color="#55617a")

    # ---- encoder stages (Swin) ----------------------------------------------
    enc_cx = enc_x + bw / 2
    keys = list(rows.keys())
    for i, k in enumerate(keys):
        y = rows[k]
        fc = C_BOTTLE if i == 3 else C_SWIN
        rbox(ax, enc_x, y, bw, bh, fc, ec=fc, lw=1.4)
        name, dim, det = enc_info[i]
        txt(ax, enc_cx, y + bh - 0.27, name, size=10.5, weight="bold", color="white")
        txt(ax, enc_cx, y + 0.27, dim, size=9, color="white")
        # detail tag to the left
        txt(ax, enc_x - 0.12, y + bh / 2, det, size=8, color="#6b7488", ha="right")

    # vertical encoder flow arrows (downsampling)
    arrow(ax, (1.55, rows["s1"]), (enc_x - 0.05, rows["s1"] + bh / 2), color=C_SWIN, lw=2.2)
    for i in range(3):
        arrow(ax, (enc_cx, rows[keys[i]]), (enc_cx, rows[keys[i + 1]] + bh),
              color=C_SWIN, lw=2.4)
        txt(ax, enc_cx + 0.95, (rows[keys[i]] + rows[keys[i + 1]] + bh) / 2,
            "patch\nmerge ↓2", size=7.5, color=C_SWIN, ha="left")

    # ---- decoder blocks (CNN) -----------------------------------------------
    dec_cx = dec_x + bw / 2
    dec_keys = ["s3", "s2", "s1"]   # decoder ascends: 192 -> 96 -> 48
    for i, k in enumerate(dec_keys):
        y = rows[k]
        rbox(ax, dec_x, y, bw, bh, C_CNN, ec=C_CNN, lw=1.4)
        name, dim, _ = dec_info[i]
        txt(ax, dec_cx, y + bh - 0.27, name, size=10.5, weight="bold", color="white")
        txt(ax, dec_cx, y + 0.27, dim, size=9, color="white")

    # bottleneck -> first decoder block
    arrow(ax, (enc_x + bw, rows["s4"] + bh / 2), (dec_x, rows["s3"] + bh / 2),
          color=C_BOTTLE, lw=2.6, rad=-0.12)
    txt(ax, (enc_x + bw + dec_x) / 2, rows["s4"] - 0.05, "bottleneck features",
        size=8.5, color=C_BOTTLE, style="italic")

    # vertical decoder flow arrows (upsampling)
    for i in range(2):
        arrow(ax, (dec_cx, rows[dec_keys[i]] + bh), (dec_cx, rows[dec_keys[i + 1]]),
              color=C_CNN, lw=2.4)
        txt(ax, dec_cx + 0.95, (rows[dec_keys[i]] + bh + rows[dec_keys[i + 1]]) / 2,
            "ConvT3d\n↑2", size=7.5, color=C_CNN, ha="left")

    # ---- cross-attention skips (encoder stage -> decoder block) -------------
    skip_pairs = [("s1", "s1"), ("s2", "s2"), ("s3", "s3")]
    xa_x = (enc_x + bw + dec_x) / 2 - 0.62
    for ek, dk in skip_pairs:
        y = rows[ek] + bh / 2
        # small cross-attn node mid-way
        rbox(ax, xa_x, y - 0.3, 1.24, 0.6, C_XATTN, ec=C_XATTN, lw=1.2, r=0.18)
        txt(ax, xa_x + 0.62, y, "Cross-Attn", size=8.3, weight="bold", color="white")
        # enc -> xa  (Key/Value)
        arrow(ax, (enc_x + bw, y), (xa_x, y), color=C_XATTN, lw=2.0)
        # xa -> dec  (Query side / fused)
        arrow(ax, (xa_x + 1.24, y), (dec_x, y), color=C_XATTN, lw=2.0)
    txt(ax, xa_x + 0.62, rows["s1"] + bh + 0.22,
        "skip = cross-attention\n(decoder Q · encoder K,V)",
        size=8.2, color=C_XATTN, weight="bold")

    # ---- upsample head + output ---------------------------------------------
    hy = rows["s1"]
    rbox(ax, 14.05, hy - 0.05, 1.5, 0.9, "#E9F4F0", ec=C_CNN, lw=1.6)
    txt(ax, 14.8, hy + 0.58, "Upsample Head", size=9.5, weight="bold", color=C_CNN)
    txt(ax, 14.8, hy + 0.28, "ConvT3d ↑4×", size=8.5, color="#55617a")
    txt(ax, 14.8, hy + 0.02, "1×1×1 conv", size=8.5, color="#55617a")
    arrow(ax, (dec_x + bw, hy + bh / 2), (14.05, hy + bh / 2), color=C_CNN, lw=2.2)

    # output cube
    rbox(ax, 14.2, hy + 1.15, 1.25, 0.62, C_HEAD, ec=C_HEAD, lw=1.4, r=0.12)
    txt(ax, 14.82, hy + 1.46, "Segmentation", size=8.6, weight="bold", color="white")
    txt(ax, 14.82, hy + 1.24, "TC · WT · ET", size=8, color="white")
    arrow(ax, (14.8, hy + 0.85), (14.8, hy + 1.15), color=C_HEAD, lw=2.0)

    # ---- auxiliary modules (bottom band) ------------------------------------
    by = 1.55
    aux = [
        (2.3, C_BOTTLE, "Temporal Attention", "baseline ↔ follow-up\nbottleneck fusion (optional)"),
        (6.55, C_SWIN, "Deep Supervision", "aux heads on decoder\nintermediates (train only)"),
        (10.8, C_HEAD, "MC Dropout", "N stochastic passes →\nvoxel uncertainty (entropy)"),
    ]
    for x, col, name, det in aux:
        rbox(ax, x, by, 2.85, 0.9, C_PANEL, ec=col, lw=1.6)
        txt(ax, x + 0.18, by + 0.6, name, size=10, weight="bold", color=col, ha="left")
        txt(ax, x + 0.18, by + 0.24, det, size=8, color="#55617a", ha="left")

    # downstream RECIST
    rbox(ax, 13.9, by, 1.85, 0.9, C_HEAD, ec=C_HEAD, lw=1.6)
    txt(ax, 14.82, by + 0.62, "RECIST + Response", size=8.8, weight="bold", color="white")
    txt(ax, 14.82, by + 0.3, "CR / PR / SD / PD", size=8.2, color="white")
    # segmentation mask -> downstream RECIST (dashed, down the right margin)
    arrow(ax, (15.45, hy + 1.46), (14.82, by + 0.9), color=C_HEAD, lw=1.8,
          rad=-0.25, ls=(0, (5, 3)))
    txt(ax, 15.62, (hy + 1.46 + by + 0.9) / 2, "lesion\nmeasurement", size=7.5,
        color=C_HEAD, ha="left", style="italic")

    # ---- legend --------------------------------------------------------------
    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_SWIN,
               markersize=13, label="Swin Transformer encoder (global context, ↓)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_CNN,
               markersize=13, label="CNN decoder (local detail / upsampling, ↑)"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_XATTN,
               markersize=13, label="Cross-attention skip fusion"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_BOTTLE,
               markersize=13, label="Bottleneck / temporal"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_HEAD,
               markersize=13, label="Output / RECIST head"),
    ]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.005),
              ncol=2, frameon=True, fontsize=9, handletextpad=0.4,
              columnspacing=1.3, borderpad=0.7).get_frame().set_edgecolor("#d4dae6")

    fig.tight_layout(pad=0.6)
    for ext in ("png", "svg"):
        fig.savefig(f"figures/architecture_diagram.{ext}", dpi=150,
                    facecolor=C_BG, bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/architecture_diagram.{png,svg}")


# ============================================================================
# FIGURE 2 — design-rationale infographic
# ============================================================================
def infographic():
    fig, ax = plt.subplots(figsize=(15, 9), dpi=150)
    fig.patch.set_facecolor(C_BG)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 9)
    ax.axis("off")

    # header band
    rbox(ax, 0.0, 8.05, 15, 0.95, C_INK, ec=C_INK, lw=0, r=0.01, shadow=False)
    txt(ax, 0.5, 8.66, "OncoSeg — Why Swin Encoder + CNN Decoder?", size=22,
        weight="bold", color="white", ha="left")
    txt(ax, 0.52, 8.27, "Transformer sees globally  •  CNN reconstructs locally  •  cross-attention bridges the two",
        size=11.5, color="#aebbd6", ha="left")

    # ---- two big role columns ------------------------------------------------
    # Swin column
    rbox(ax, 0.4, 4.6, 6.9, 3.05, "#EEF2FA", ec=C_SWIN, lw=2.0)
    txt(ax, 0.75, 7.28, "SWIN TRANSFORMER  =  ENCODER", size=13.5, weight="bold",
        color=C_SWIN, ha="left")
    txt(ax, 0.75, 6.96, "“the eyes & brain” — extract multi-scale global features",
        size=10, color="#55617a", ha="left", style="italic")
    swin_pts = [
        "Shifted-window self-attention → long-range context",
        "Windowing keeps 3D attention memory-feasible",
        "4 hierarchical stages: 48→96→192→384 ch,  H/4→H/32",
        "Captures whole-tumor / edema relationships a CNN's",
        "   limited receptive field would miss",
    ]
    for i, s in enumerate(swin_pts):
        txt(ax, 0.78, 6.55 - i * 0.36, ("•  " if not s.startswith("   ") else "    ") + s.strip(),
            size=10, color=C_INK, ha="left")

    # CNN column
    rbox(ax, 7.7, 4.6, 6.9, 3.05, "#EAF5F1", ec=C_CNN, lw=2.0)
    txt(ax, 8.05, 7.28, "CNN  =  DECODER", size=13.5, weight="bold", color=C_CNN, ha="left")
    txt(ax, 8.05, 6.96, "“the hands” — rebuild voxel-level masks at full resolution",
        size=10, color="#55617a", ha="left", style="italic")
    cnn_pts = [
        "Transposed-conv upsampling → precise boundaries",
        "InstanceNorm + LeakyReLU refine local structure",
        "Dense per-voxel prediction, parameter/memory-efficient",
        "Self-attention here would explode (tokens = H·W·D)",
        "Recovers the 4× patch-embed downsampling at the head",
    ]
    for i, s in enumerate(cnn_pts):
        txt(ax, 8.08, 6.55 - i * 0.36, "•  " + s, size=10, color=C_INK, ha="left")

    # connecting arrows + cross-attn chip in the middle
    arrow(ax, (7.32, 6.1), (8.55, 6.1), color=C_XATTN, lw=2.6, rad=0.0)
    arrow(ax, (8.55, 5.2), (7.32, 5.2), color=C_XATTN, lw=2.6, rad=0.0)
    rbox(ax, 6.55, 5.4, 1.9, 0.7, C_XATTN, ec=C_XATTN, lw=1.4, r=0.2)
    txt(ax, 7.5, 5.75, "Cross-Attention", size=9.2, weight="bold", color="white")

    # ---- cross-attention explainer strip ------------------------------------
    rbox(ax, 0.4, 3.05, 14.2, 1.25, C_PANEL, ec=C_XATTN, lw=1.8)
    txt(ax, 0.72, 3.95, "CROSS-ATTENTION SKIP  (the project's twist)", size=12,
        weight="bold", color=C_XATTN, ha="left")
    txt(ax, 0.72, 3.55,
        "Plain U-Nets concat encoder→decoder. OncoSeg instead lets the decoder QUERY the encoder:",
        size=10, color=C_INK, ha="left")
    txt(ax, 0.72, 3.24,
        "decoder feature = Query    encoder feature = Key / Value   →   selectively pull the relevant global cues; aligns Swin's semantics with CNN's geometry (ablated by  no_xattn).",
        size=9.6, color="#55617a", ha="left")

    # ---- stats row -----------------------------------------------------------
    stats = [
        ("Best Dice", "0.797", "MSD Brain, 50 ep", C_SWIN),
        ("WT / TC / ET", "0.85 / 0.79 / 0.75", "per-region Dice", C_CNN),
        ("vs UNet3D", "5× fewer params", "wins all regions", C_BOTTLE),
        ("Calibration", "ECE 0.010", "MC-Dropout, reliable", C_HEAD),
        ("Validation", "LUMIERE", "longitudinal RECIST", C_XATTN),
    ]
    sw = 2.74
    for i, (k, v, sub, col) in enumerate(stats):
        x = 0.4 + i * (sw + 0.18)
        rbox(ax, x, 1.35, sw, 1.4, "white", ec=col, lw=2.0)
        txt(ax, x + sw / 2, 2.4, k, size=10, weight="bold", color=col)
        txt(ax, x + sw / 2, 2.02, v, size=15, weight="bold", color=C_INK)
        txt(ax, x + sw / 2, 1.6, sub, size=8.5, color="#7a8499")

    # ---- footer pipeline -----------------------------------------------------
    flow = ["4-modal MRI", "Swin encode", "X-attn skips", "CNN decode",
            "TC/WT/ET mask", "RECIST", "CR/PR/SD/PD"]
    fx = 0.55
    fy = 0.6
    cols = [C_SWIN, C_SWIN, C_XATTN, C_CNN, C_HEAD, C_HEAD, C_HEAD]
    for i, (label, col) in enumerate(zip(flow, cols)):
        w = 1.78
        rbox(ax, fx, fy, w, 0.55, col, ec=col, lw=0, r=0.16, shadow=False)
        txt(ax, fx + w / 2, fy + 0.275, label, size=8.8, weight="bold", color="white")
        if i < len(flow) - 1:
            arrow(ax, (fx + w, fy + 0.275), (fx + w + 0.18, fy + 0.275),
                  color="#9aa4b6", lw=1.6)
        fx += w + 0.18

    fig.tight_layout(pad=0.5)
    for ext in ("png", "svg"):
        fig.savefig(f"figures/infographic.{ext}", dpi=150, facecolor=C_BG,
                    bbox_inches="tight")
    plt.close(fig)
    print("wrote figures/infographic.{png,svg}")


if __name__ == "__main__":
    _ = fm  # keep import (font discovery side-effect on some setups)
    architecture()
    infographic()
