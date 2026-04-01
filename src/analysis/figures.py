"""Generate publication-quality figures for OncoSeg results."""

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
    })
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


MODEL_COLORS = {
    "oncoseg": "#e74c3c",
    "unet3d": "#3498db",
    "swin_unetr": "#2ecc71",
    "unetr": "#9b59b6",
    "oncoseg_concat_skip": "#e67e22",
    "oncoseg_no_ds": "#1abc9c",
}


class FigureGenerator:
    """Generate all figures for the OncoSeg paper/report.

    Usage:
        fig_gen = FigureGenerator(output_dir="figures/")
        fig_gen.training_curves(histories)
        fig_gen.dice_comparison_bar(evaluations)
        fig_gen.ablation_chart(ablation_results)
    """

    def __init__(self, output_dir: str | Path = "figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def training_curves(
        self, histories: dict[str, dict], val_interval: int = 5
    ) -> Path | None:
        """Plot training loss and validation Dice curves.

        Args:
            histories: {model_name: {"train_loss": [...], "val_dice_mean": [...], ...}}
            val_interval: Epochs between validations.

        Returns:
            Path to saved figure.
        """
        if not HAS_MATPLOTLIB:
            return None

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        for name, hist in histories.items():
            color = MODEL_COLORS.get(name, "#333333")

            # Training loss
            if "train_loss" in hist:
                axes[0].plot(
                    hist["train_loss"],
                    label=name,
                    color=color,
                    linewidth=2,
                )

            # Validation Dice
            if "val_dice_mean" in hist and hist["val_dice_mean"]:
                n_vals = len(hist["val_dice_mean"])
                val_epochs = list(range(val_interval, val_interval * n_vals + 1, val_interval))
                best = hist.get("best_dice", max(hist["val_dice_mean"]))
                axes[1].plot(
                    val_epochs,
                    hist["val_dice_mean"],
                    label=f"{name} (best: {best:.4f})",
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=3,
                )

        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Training Loss")
        axes[0].set_title("Training Loss")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Mean Dice Score")
        axes[1].set_title("Validation Dice Score")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle("OncoSeg — Training on MSD Brain Tumor Data", fontsize=16, fontweight="bold")
        plt.tight_layout()

        save_path = self.output_dir / "training_curves.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def dice_comparison_bar(self, evaluations: dict[str, dict]) -> Path | None:
        """Bar chart comparing Dice scores across models and regions.

        Args:
            evaluations: {model_name: {"dice_ET": ..., "dice_TC": ..., "dice_WT": ...}}
        """
        if not HAS_MATPLOTLIB:
            return None

        models = sorted(evaluations.keys())
        regions = ["ET", "TC", "WT"]
        x = np.arange(len(regions))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(models):
            res = evaluations[name]
            scores = [res.get(f"dice_{r}", 0) for r in regions]
            color = MODEL_COLORS.get(name, "#333333")
            bars = ax.bar(x + i * width, scores, width, label=name, color=color, alpha=0.85)

            for bar, score in zip(bars, scores):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{score:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        ax.set_xlabel("Tumor Region")
        ax.set_ylabel("Dice Score")
        ax.set_title("Dice Score Comparison by Tumor Region")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(["Enhancing\nTumor (ET)", "Tumor\nCore (TC)", "Whole\nTumor (WT)"])
        ax.set_ylim(0, 1.0)
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = self.output_dir / "dice_comparison.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def hd95_comparison_bar(self, evaluations: dict[str, dict]) -> Path | None:
        """Bar chart comparing HD95 scores (lower is better)."""
        if not HAS_MATPLOTLIB:
            return None

        models = sorted(evaluations.keys())
        regions = ["ET", "TC", "WT"]
        x = np.arange(len(regions))
        width = 0.8 / len(models)

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, name in enumerate(models):
            res = evaluations[name]
            scores = [res.get(f"hd95_{r}", 0) for r in regions]
            color = MODEL_COLORS.get(name, "#333333")
            ax.bar(x + i * width, scores, width, label=name, color=color, alpha=0.85)

        ax.set_xlabel("Tumor Region")
        ax.set_ylabel("HD95 (mm) — lower is better")
        ax.set_title("Hausdorff Distance 95% Comparison")
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(["Enhancing\nTumor (ET)", "Tumor\nCore (TC)", "Whole\nTumor (WT)"])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        save_path = self.output_dir / "hd95_comparison.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path

    def ablation_chart(self, ablation_results: dict[str, float]) -> Path | None:
        """Horizontal bar chart for ablation study results.

        Args:
            ablation_results: {"OncoSeg (full)": 0.85, "No cross-attn": 0.81, ...}
        """
        if not HAS_MATPLOTLIB:
            return None

        names = list(ablation_results.keys())
        scores = list(ablation_results.values())

        fig, ax = plt.subplots(figsize=(10, max(4, len(names) * 0.8)))

        colors = ["#e74c3c" if i == 0 else "#95a5a6" for i in range(len(names))]
        bars = ax.barh(names, scores, color=colors, alpha=0.85)

        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}",
                va="center",
                fontsize=10,
            )

        ax.set_xlabel("Mean Dice Score")
        ax.set_title("Ablation Study — Component Contribution")
        ax.set_xlim(0, 1.0)
        ax.grid(True, alpha=0.3, axis="x")
        ax.invert_yaxis()

        plt.tight_layout()
        save_path = self.output_dir / "ablation_study.png"
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        return save_path
