"""CLI entry point for `oncoseg-serve`."""

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="oncoseg-serve",
        description="Start the OncoSeg FastAPI inference service.",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--checkpoint",
        default=os.environ.get("ONCOSEG_CHECKPOINT"),
        help="Path to .pth checkpoint (or set ONCOSEG_CHECKPOINT)",
    )
    parser.add_argument(
        "--model-source",
        default=os.environ.get("ONCOSEG_MODEL_SOURCE", "train_all"),
        choices=["train_all", "src"],
    )
    parser.add_argument(
        "--roi-size",
        default=os.environ.get("ONCOSEG_ROI_SIZE", "128,128,128"),
        help="Sliding-window ROI, comma-separated H,W,D",
    )
    parser.add_argument("--sw-batch-size", type=int, default=2)
    parser.add_argument("--mc-samples", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    if args.checkpoint:
        os.environ["ONCOSEG_CHECKPOINT"] = args.checkpoint
    os.environ["ONCOSEG_MODEL_SOURCE"] = args.model_source
    os.environ["ONCOSEG_ROI_SIZE"] = args.roi_size
    os.environ["ONCOSEG_SW_BATCH_SIZE"] = str(args.sw_batch_size)
    os.environ["ONCOSEG_MC_SAMPLES"] = str(args.mc_samples)

    try:
        import uvicorn
    except ImportError:
        sys.stderr.write(
            "uvicorn is not installed. Install the serve extras:\n"
            "    pip install -e '.[serve]'\n"
        )
        sys.exit(1)

    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
