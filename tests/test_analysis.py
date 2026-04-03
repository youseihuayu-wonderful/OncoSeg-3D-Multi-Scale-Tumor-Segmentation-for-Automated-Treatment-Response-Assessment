"""Unit tests for the analysis toolkit."""

import json

import numpy as np
import pytest

from src.analysis.failure_analyzer import FailureAnalyzer
from src.analysis.result_analyzer import ResultAnalyzer


class TestResultAnalyzer:
    """Test result comparison and statistical analysis."""

    @pytest.fixture
    def analyzer(self, tmp_path):
        analyzer = ResultAnalyzer()

        # Create fake training histories
        for name, best_dice, best_epoch in [
            ("oncoseg", 0.85, 90),
            ("unet3d", 0.78, 70),
            ("swin_unetr", 0.82, 85),
        ]:
            hist = {
                "train_loss": [1.0 - i * 0.008 for i in range(100)],
                "val_dice_mean": [0.3 + i * 0.005 for i in range(20)],
                "best_dice": best_dice,
                "best_epoch": best_epoch,
            }
            path = tmp_path / f"{name}_history.json"
            with open(path, "w") as f:
                json.dump(hist, f)
            analyzer.load_history(name, path)

        # Create fake evaluation results
        n_subjects = 20
        rng = np.random.RandomState(42)
        for name, base_dice in [("oncoseg", 0.85), ("unet3d", 0.78), ("swin_unetr", 0.82)]:
            per_subject = rng.normal(base_dice, 0.05, size=(n_subjects, 3)).clip(0, 1)
            analyzer.load_eval(
                name,
                {
                    "dice_ET": float(per_subject[:, 0].mean()),
                    "dice_TC": float(per_subject[:, 1].mean()),
                    "dice_WT": float(per_subject[:, 2].mean()),
                    "dice_mean": float(per_subject.mean()),
                    "hd95_ET": 5.0 - base_dice * 3,
                    "hd95_TC": 4.0 - base_dice * 2,
                    "hd95_WT": 3.0 - base_dice * 1.5,
                    "hd95_mean": 4.0 - base_dice * 2,
                    "per_subject_dice": per_subject.tolist(),
                },
            )

        return analyzer

    def test_best_dice_summary(self, analyzer):
        summary = analyzer.best_dice_summary()
        assert "oncoseg" in summary
        assert "unet3d" in summary
        assert "0.85" in summary

    def test_comparison_table(self, analyzer):
        table = analyzer.comparison_table()
        assert "oncoseg" in table
        assert "Dice_ET" in table

    def test_significance_tests(self, analyzer):
        results = analyzer.significance_tests("oncoseg")
        assert "oncoseg vs unet3d" in results
        assert "ET" in results
        assert "p=" in results

    def test_convergence_analysis(self, analyzer):
        analysis = analyzer.convergence_analysis()
        assert "oncoseg" in analysis
        assert "Final loss" in analysis

    def test_per_region_breakdown(self, analyzer):
        breakdown = analyzer.per_region_breakdown()
        assert "ET" in breakdown
        assert "Winner" in breakdown

    def test_load_eval_from_json(self, tmp_path):
        analyzer = ResultAnalyzer()
        data = {
            "model_a": {"dice_ET": 0.8, "dice_TC": 0.85, "dice_WT": 0.9, "dice_mean": 0.85},
            "model_b": {"dice_ET": 0.7, "dice_TC": 0.75, "dice_WT": 0.8, "dice_mean": 0.75},
        }
        path = tmp_path / "eval.json"
        with open(path, "w") as f:
            json.dump(data, f)
        analyzer.load_eval_from_json(path)
        assert "model_a" in analyzer.evaluations
        assert "model_b" in analyzer.evaluations


class TestFailureAnalyzer:
    """Test failure case analysis."""

    @pytest.fixture
    def analyzer(self):
        fa = FailureAnalyzer(dice_threshold=0.5)

        # Add subjects with varying tumor sizes and performance
        # Small tumor, low dice (failure case)
        gt_small = np.zeros((32, 32, 32), dtype=np.uint8)
        gt_small[14:18, 14:18, 14:18] = 1  # 64 voxels = small
        pred_small = np.zeros((32, 32, 32), dtype=np.uint8)
        pred_small[16:20, 16:20, 16:20] = 1  # Shifted prediction
        fa.add_subject("sub_001", pred_small, gt_small, {"ET": 0.3, "TC": 0.4, "WT": 0.35})

        # Large tumor, high dice (good case)
        gt_large = np.zeros((64, 64, 64), dtype=np.uint8)
        gt_large[10:50, 10:50, 10:50] = 1  # 64000 voxels = large
        pred_large = np.zeros((64, 64, 64), dtype=np.uint8)
        pred_large[11:49, 11:49, 11:49] = 1
        fa.add_subject("sub_002", pred_large, gt_large, {"ET": 0.9, "TC": 0.88, "WT": 0.92})

        # Medium tumor, moderate dice
        gt_med = np.zeros((32, 32, 32), dtype=np.uint8)
        gt_med[5:20, 5:20, 5:20] = 1  # 3375 voxels = medium
        pred_med = np.zeros((32, 32, 32), dtype=np.uint8)
        pred_med[7:22, 7:22, 7:22] = 1
        fa.add_subject("sub_003", pred_med, gt_med, {"ET": 0.65, "TC": 0.7, "WT": 0.68})

        # No tumor case
        gt_empty = np.zeros((32, 32, 32), dtype=np.uint8)
        pred_empty = np.zeros((32, 32, 32), dtype=np.uint8)
        fa.add_subject("sub_004", pred_empty, gt_empty, {"ET": 1.0, "TC": 1.0, "WT": 1.0})

        return fa

    def test_subject_count(self, analyzer):
        assert len(analyzer.subjects) == 4

    def test_failure_report(self, analyzer):
        report = analyzer.failure_report(top_n=5)
        assert "sub_001" in report  # Worst case should appear
        assert "Failure" in report or "failure" in report

    def test_size_stratified(self, analyzer):
        analysis = analyzer.size_stratified_analysis()
        assert "small" in analysis
        assert "medium" in analysis
        assert "large" in analysis

    def test_segmentation_bias(self, analyzer):
        analysis = analyzer.segmentation_bias_analysis()
        assert (
            "Over-segmentation" in analysis
            or "Under-segmentation" in analysis
            or "Balanced" in analysis
        )

    def test_size_categorization(self, analyzer):
        categories = {s["size_category"] for s in analyzer.subjects}
        assert "small" in categories
        assert "large" in categories
        assert "medium" in categories
        assert "no_tumor" in categories

    def test_failure_types(self, analyzer):
        types = {s["failure_type"] for s in analyzer.subjects}
        assert "failure" in types  # sub_001 has mean dice 0.35
        assert "good" in types  # sub_002 has mean dice 0.9


class TestFigureGenerator:
    """Test figure generation (creates files, doesn't check visual quality)."""

    def test_training_curves(self, tmp_path):
        from src.analysis.figures import FigureGenerator

        fg = FigureGenerator(output_dir=tmp_path)
        histories = {
            "oncoseg": {
                "train_loss": [1.0 - i * 0.01 for i in range(50)],
                "val_dice_mean": [0.3 + i * 0.01 for i in range(10)],
                "best_dice": 0.39,
            },
            "unet3d": {
                "train_loss": [1.1 - i * 0.01 for i in range(50)],
                "val_dice_mean": [0.25 + i * 0.01 for i in range(10)],
                "best_dice": 0.34,
            },
        }
        path = fg.training_curves(histories, val_interval=5)
        assert path is not None
        assert path.exists()
        assert path.suffix == ".png"

    def test_dice_comparison_bar(self, tmp_path):
        from src.analysis.figures import FigureGenerator

        fg = FigureGenerator(output_dir=tmp_path)
        evaluations = {
            "oncoseg": {"dice_ET": 0.82, "dice_TC": 0.85, "dice_WT": 0.90},
            "unet3d": {"dice_ET": 0.75, "dice_TC": 0.78, "dice_WT": 0.85},
        }
        path = fg.dice_comparison_bar(evaluations)
        assert path is not None
        assert path.exists()

    def test_ablation_chart(self, tmp_path):
        from src.analysis.figures import FigureGenerator

        fg = FigureGenerator(output_dir=tmp_path)
        ablation = {
            "OncoSeg (full)": 0.85,
            "No cross-attention": 0.81,
            "No deep supervision": 0.83,
        }
        path = fg.ablation_chart(ablation)
        assert path is not None
        assert path.exists()
