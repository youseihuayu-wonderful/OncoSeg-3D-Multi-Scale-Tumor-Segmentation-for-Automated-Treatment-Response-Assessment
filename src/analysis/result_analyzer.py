"""Analyze and compare training results across models."""

import json
from pathlib import Path

import numpy as np
from scipy import stats


class ResultAnalyzer:
    """Load, compare, and statistically test model results.

    Usage:
        analyzer = ResultAnalyzer()
        analyzer.load_history("oncoseg", "/content/oncoseg_history.json")
        analyzer.load_history("unet3d", "/content/unet3d_history.json")
        analyzer.load_eval("oncoseg", oncoseg_eval_dict)
        analyzer.load_eval("unet3d", unet3d_eval_dict)
        print(analyzer.comparison_table())
        print(analyzer.significance_tests("oncoseg"))
    """

    REGIONS = ["ET", "TC", "WT"]

    def __init__(self):
        self.histories: dict[str, dict] = {}
        self.evaluations: dict[str, dict] = {}

    def load_history(self, model_name: str, path: str | Path):
        """Load training history from JSON file."""
        with open(path) as f:
            self.histories[model_name] = json.load(f)

    def load_eval(self, model_name: str, results: dict):
        """Store evaluation results dict for a model."""
        self.evaluations[model_name] = results

    def load_eval_from_json(self, path: str | Path):
        """Load evaluation results from saved JSON (all models)."""
        with open(path) as f:
            data = json.load(f)
        for name, results in data.items():
            self.evaluations[name] = results

    def best_dice_summary(self) -> str:
        """Print best validation Dice for all models."""
        lines = ["Model                Best Dice    Best Epoch", "─" * 50]
        for name, hist in sorted(self.histories.items()):
            best = hist.get("best_dice", 0)
            epoch = hist.get("best_epoch", "?")
            lines.append(f"{name:20s} {best:.4f}       {epoch}")
        return "\n".join(lines)

    def comparison_table(self) -> str:
        """Generate full comparison table across all evaluated models."""
        lines = [
            "Model              Dice_ET  Dice_TC  Dice_WT  Dice_Mean  HD95_ET  HD95_TC  HD95_WT  HD95_Mean",
            "─" * 100,
        ]
        for name, res in sorted(self.evaluations.items()):
            lines.append(
                f"{name:18s} "
                f"{res.get('dice_ET', 0):.4f}   "
                f"{res.get('dice_TC', 0):.4f}   "
                f"{res.get('dice_WT', 0):.4f}   "
                f"{res.get('dice_mean', 0):.4f}     "
                f"{res.get('hd95_ET', 0):6.2f}   "
                f"{res.get('hd95_TC', 0):6.2f}   "
                f"{res.get('hd95_WT', 0):6.2f}   "
                f"{res.get('hd95_mean', 0):6.2f}"
            )
        return "\n".join(lines)

    def significance_tests(self, target_model: str, alternative: str = "greater") -> str:
        """Run Wilcoxon signed-rank tests: target vs all other models.

        Args:
            target_model: The model to compare against others.
            alternative: "greater" tests if target > baseline.

        Returns:
            Formatted significance test results.
        """
        if target_model not in self.evaluations:
            return f"No evaluation data for {target_model}"

        target_dice = self.evaluations[target_model].get("per_subject_dice")
        if target_dice is None:
            return "No per-subject dice scores available for significance testing"

        if isinstance(target_dice, list):
            target_dice = np.array(target_dice)

        lines = [
            f"Statistical Tests: {target_model} vs baselines (Wilcoxon signed-rank, {alternative})",
            "─" * 80,
        ]

        for baseline_name, baseline_res in sorted(self.evaluations.items()):
            if baseline_name == target_model:
                continue

            baseline_dice = baseline_res.get("per_subject_dice")
            if baseline_dice is None:
                continue
            if isinstance(baseline_dice, list):
                baseline_dice = np.array(baseline_dice)

            lines.append(f"\n  {target_model} vs {baseline_name}:")

            for i, region in enumerate(self.REGIONS):
                target_scores = target_dice[:, i]
                baseline_scores = baseline_dice[:, i]
                delta = np.mean(target_scores - baseline_scores)

                try:
                    stat, p_value = stats.wilcoxon(
                        target_scores, baseline_scores, alternative=alternative
                    )
                    sig = (
                        "***"
                        if p_value < 0.001
                        else "**"
                        if p_value < 0.01
                        else "*"
                        if p_value < 0.05
                        else "ns"
                    )
                except ValueError:
                    p_value = 1.0
                    sig = "ns (identical)"

                lines.append(f"    {region}: p={p_value:.4f} {sig:>4s}  Δ={delta:+.4f}")

        return "\n".join(lines)

    def convergence_analysis(self) -> str:
        """Analyze training convergence for each model."""
        lines = ["Convergence Analysis", "─" * 60]

        for name, hist in sorted(self.histories.items()):
            losses = hist.get("train_loss", [])
            if not losses:
                continue

            final_loss = losses[-1]
            min_loss = min(losses)
            min_epoch = losses.index(min_loss) + 1

            # Check for overfitting: loss increasing in last 20%
            n = len(losses)
            last_20 = losses[int(n * 0.8) :]
            overfitting = last_20[-1] > last_20[0] if len(last_20) > 1 else False

            lines.append(f"\n  {name}:")
            lines.append(f"    Final loss:     {final_loss:.4f}")
            lines.append(f"    Min loss:       {min_loss:.4f} (epoch {min_epoch})")
            lines.append(
                f"    Overfitting:    {'YES — loss rising in last 20%' if overfitting else 'No'}"
            )
            lines.append(
                f"    Best Dice:      {hist.get('best_dice', 0):.4f} (epoch {hist.get('best_epoch', '?')})"
            )

        return "\n".join(lines)

    def per_region_breakdown(self) -> str:
        """Detailed per-region analysis showing which model wins where."""
        lines = ["Per-Region Winners", "─" * 60]

        for region in self.REGIONS:
            dice_key = f"dice_{region}"
            best_name = None
            best_dice = -1

            for name, res in self.evaluations.items():
                val = res.get(dice_key, 0)
                if val > best_dice:
                    best_dice = val
                    best_name = name

            lines.append(f"\n  {region}:")
            lines.append(f"    Winner: {best_name} ({best_dice:.4f})")
            for name, res in sorted(self.evaluations.items()):
                val = res.get(dice_key, 0)
                gap = val - best_dice
                marker = " ★" if name == best_name else ""
                lines.append(f"      {name:18s} {val:.4f}  ({gap:+.4f}){marker}")

        return "\n".join(lines)
