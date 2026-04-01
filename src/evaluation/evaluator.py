"""Model evaluation with statistical testing and result persistence."""

import json
import logging
from pathlib import Path

import hydra
import torch
import torch.nn as nn
from monai.data import DataLoader
from monai.inferers import sliding_window_inference
from omegaconf import DictConfig
from tqdm import tqdm

from src.evaluation.metrics import SegmentationMetrics

logger = logging.getLogger(__name__)


class Evaluator:
    """Evaluate trained models and save reproducible results."""

    def __init__(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        cfg: DictConfig,
    ):
        self.model = model
        self.test_loader = test_loader
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self.metrics = SegmentationMetrics()

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run full evaluation on test set."""
        self.model.eval()
        self.metrics.reset()

        for batch in tqdm(self.test_loader, desc="Evaluating"):
            images = torch.cat(
                [batch[k] for k in ["t1n", "t1c", "t2w", "t2f"]], dim=1
            ).to(self.device)
            labels = batch["label"].to(self.device)

            preds = sliding_window_inference(
                inputs=images,
                roi_size=tuple(self.cfg.data.roi_size),
                sw_batch_size=self.cfg.training.sw_batch_size,
                predictor=lambda x: self.model(x)["pred"],
                overlap=0.5,
            )

            preds = (torch.softmax(preds, dim=1) > 0.5).float()
            self.metrics.update(preds, labels)

        results = self.metrics.compute()
        logger.info("\n" + self.metrics.summary())

        return results

    def evaluate_multi_seed(self, seeds: list[int], checkpoint_dir: str) -> dict:
        """Evaluate across multiple random seeds for statistical validity."""
        all_results = []

        for seed in seeds:
            ckpt_path = Path(checkpoint_dir) / f"seed_{seed}" / "best.pth"
            if not ckpt_path.exists():
                logger.warning(f"Checkpoint not found: {ckpt_path}")
                continue

            checkpoint = torch.load(ckpt_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])

            results = self.evaluate()
            results["seed"] = seed
            all_results.append(results)

        # Compute mean ± std
        import numpy as np

        summary = {}
        metric_keys = [k for k in all_results[0] if k != "seed"]
        for key in metric_keys:
            values = [r[key] for r in all_results]
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))

        summary["num_seeds"] = len(all_results)
        summary["per_seed"] = all_results

        return summary

    def save_results(self, results: dict, save_path: str):
        """Save evaluation results to JSON for reproducibility."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info(f"Results saved to {save_path}")


@hydra.main(version_base=None, config_path="../../configs", config_name="config")
def main(cfg: DictConfig):
    """Entry point for evaluation."""
    from src.data import BraTSDataset, get_val_transforms
    from src.models import OncoSeg

    test_ds = BraTSDataset(
        root_dir=cfg.data.root_dir,
        split="test",
        transform=get_val_transforms(),
    ).get_dataset()
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)

    model = OncoSeg(
        in_channels=cfg.model.in_channels,
        num_classes=cfg.model.num_classes,
        embed_dim=cfg.model.embed_dim,
    )

    evaluator = Evaluator(model, test_loader, cfg)
    results = evaluator.evaluate()
    evaluator.save_results(results, cfg.evaluation.results_path)


if __name__ == "__main__":
    main()
