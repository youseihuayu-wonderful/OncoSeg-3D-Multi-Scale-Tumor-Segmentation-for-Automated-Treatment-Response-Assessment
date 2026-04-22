"""Train and evaluate OncoSeg ablation variants.

Trains 4 ablation variants to isolate the contribution of each
architectural component:

  oncoseg_no_xattn  — additive skips instead of cross-attention
  oncoseg_no_ds     — deep supervision disabled
  oncoseg_no_mcdrop — MC Dropout set to 0
  oncoseg_small     — half embed_dim (24 vs 48)

Usage (Colab / CUDA):
    python scripts/run_ablation.py --epochs 100 --device cuda

Usage (local dry-run):
    python scripts/run_ablation.py --epochs 2 --device cpu --roi-size 48 --embed-dim 24
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_all import (  # noqa: E402
    SAVE_DIR,
    build_data,
    build_model,
    logger,
    train_model,
)

ABLATION_VARIANTS = [
    "oncoseg_no_xattn",
    "oncoseg_no_ds",
    "oncoseg_no_mcdrop",
    "oncoseg_small",
]


def count_params(name: str, roi_size: tuple, embed_dim: int) -> float:
    model = build_model(name, roi_size, embed_dim=embed_dim)
    n = sum(p.numel() for p in model.parameters()) / 1e6
    del model
    return n


def main() -> int:
    parser = argparse.ArgumentParser(description="OncoSeg ablation study")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--roi-size", type=int, default=96)
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--variants", type=str, nargs="+",
                        default=ABLATION_VARIANTS,
                        help="Which variants to train")
    args = parser.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(42)
    np.random.seed(42)

    roi_size = (args.roi_size, args.roi_size, args.roi_size)
    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Variants: {args.variants}")
    logger.info(f"Epochs: {args.epochs}, ROI: {args.roi_size}, embed_dim: {args.embed_dim}")

    # Print param counts
    logger.info("\nParameter counts:")
    baseline_params = count_params("oncoseg", roi_size, args.embed_dim)
    logger.info(f"  {'oncoseg (baseline)':25s}: {baseline_params:.3f}M")
    for v in args.variants:
        p = count_params(v, roi_size, args.embed_dim)
        delta = p - baseline_params
        logger.info(f"  {v:25s}: {p:.3f}M ({delta:+.3f}M)")

    # Load data
    logger.info("\nLoading data...")
    train_ds, val_ds, n_train, n_val = build_data(roi_size)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1,
                            shuffle=False, num_workers=0)
    logger.info(f"Train: {n_train} | Val: {n_val}")

    # Train each variant
    all_histories = {}
    start = time.time()

    for variant in args.variants:
        best_path = SAVE_DIR / f"{variant}_best.pth"
        ckpt_path = SAVE_DIR / f"{variant}_checkpoint.pth"
        history_path = SAVE_DIR / f"{variant}_history.json"

        if best_path.exists() and not ckpt_path.exists() and history_path.exists():
            logger.info(f"\n{variant} already completed - loading saved history")
            with open(history_path) as f:
                all_histories[variant] = json.load(f)
            continue

        history = train_model(
            variant, train_loader, val_loader, device, roi_size,
            max_epochs=args.epochs, lr=args.lr,
            val_interval=args.val_interval, embed_dim=args.embed_dim,
        )
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        all_histories[variant] = history

    total_time = time.time() - start

    # Save ablation results
    results = {
        "config": {
            "epochs": args.epochs,
            "roi_size": args.roi_size,
            "embed_dim": args.embed_dim,
            "device": str(device),
            "total_time_minutes": total_time / 60,
        },
        "variants": {},
    }
    for name, hist in all_histories.items():
        results["variants"][name] = {
            "best_dice": hist["best_dice"],
            "best_epoch": hist["best_epoch"],
            "params_M": count_params(name, roi_size, args.embed_dim),
        }

    out_path = SAVE_DIR / "ablation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    # Print summary
    logger.info(f"\n{'='*60}")
    logger.info("ABLATION RESULTS SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Total time: {total_time/60:.1f} minutes")
    for name, hist in all_histories.items():
        logger.info(
            f"  {name:25s}: Best Dice = {hist['best_dice']:.4f} "
            f"(epoch {hist['best_epoch']})"
        )
    logger.info(f"\nResults saved to: {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
