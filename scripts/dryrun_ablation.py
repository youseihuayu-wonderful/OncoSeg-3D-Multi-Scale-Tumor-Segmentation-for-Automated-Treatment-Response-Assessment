"""Local smoke test for the OncoSeg ablation harness.

Instantiates the baseline OncoSeg plus all 4 ablation variants, runs a single
forward pass on a small dummy volume, and verifies that each variant compiles,
produces the expected output shape, and reports a sane parameter count.

This is a *shape and compile* check, not a training/eval substitute. It exists
so that any Python-level bug in a variant is caught locally on M1 in seconds —
before burning Colab T4 compute on a run that crashes at epoch 0.

Usage:
    python scripts/dryrun_ablation.py
    python scripts/dryrun_ablation.py --roi-size 48 --embed-dim 24
"""

import argparse
import sys
from pathlib import Path

import torch

# Make train_all importable when running from the repo root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from train_all import build_model  # noqa: E402

VARIANTS = [
    "oncoseg",
    "oncoseg_no_xattn",
    "oncoseg_no_ds",
    "oncoseg_no_mcdrop",
    "oncoseg_small",
]


def count_params(model: torch.nn.Module) -> float:
    """Return total parameter count in millions."""
    return sum(p.numel() for p in model.parameters()) / 1e6


def run_variant(name: str, roi: int, embed_dim: int, device: torch.device) -> dict:
    """Build, count params, forward-pass, verify output shape."""
    roi_size = (roi, roi, roi)
    model = build_model(name, roi_size, embed_dim=embed_dim).to(device)
    n_params = count_params(model)

    # Eval mode so deep_sup outputs are not produced (forward returns only "pred").
    model.eval()
    x = torch.randn(1, 4, roi, roi, roi, device=device)

    with torch.no_grad():
        out = model(x)

    assert "pred" in out, f"{name}: forward did not return 'pred' key"
    pred = out["pred"]
    expected_shape = (1, 3, roi, roi, roi)
    assert tuple(pred.shape) == expected_shape, (
        f"{name}: bad output shape {tuple(pred.shape)}, expected {expected_shape}"
    )

    return {"name": name, "params_M": n_params, "out_shape": tuple(pred.shape)}


def main() -> int:
    parser = argparse.ArgumentParser(description="OncoSeg ablation dry-run")
    parser.add_argument("--roi-size", type=int, default=64,
                        help="Cubic ROI side length for the dummy input volume")
    parser.add_argument("--embed-dim", type=int, default=24,
                        help="Baseline embed_dim — matches train_all.py default")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}  |  ROI: {args.roi_size}^3  |  embed_dim: {args.embed_dim}")
    print("-" * 64)

    results = []
    failures = []
    for name in VARIANTS:
        try:
            r = run_variant(name, args.roi_size, args.embed_dim, device)
            results.append(r)
            print(f"  PASS  {name:22s}  params={r['params_M']:7.3f}M  out={r['out_shape']}")
        except Exception as e:
            failures.append((name, e))
            print(f"  FAIL  {name:22s}  {type(e).__name__}: {e}")

    print("-" * 64)
    if failures:
        print(f"{len(failures)} variant(s) FAILED — fix before training")
        return 1

    # Sanity comparisons relative to baseline
    baseline = next(r for r in results if r["name"] == "oncoseg")
    print(f"\nBaseline oncoseg: {baseline['params_M']:.3f}M params")
    for r in results:
        if r["name"] == "oncoseg":
            continue
        delta = r["params_M"] - baseline["params_M"]
        sign = "+" if delta >= 0 else "-"
        pct = 100.0 * delta / baseline["params_M"]
        print(f"  {r['name']:22s} {sign}{abs(delta):6.3f}M  ({pct:+6.1f}%)")

    print("\nAll variants compiled and produced correct output shape.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
