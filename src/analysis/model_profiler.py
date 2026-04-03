"""Model profiling: parameter counts, memory estimation, and inference speed.

Provides a structured comparison of model complexity for the paper's
efficiency table.
"""

import time

import torch
import torch.nn as nn


def count_parameters(model: nn.Module) -> dict[str, int]:
    """Count total, trainable, and frozen parameters.

    Returns:
        Dict with "total", "trainable", "frozen" parameter counts.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        "total": total,
        "trainable": trainable,
        "frozen": total - trainable,
    }


def estimate_memory_mb(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 4, 128, 128, 128),
    dtype_bytes: int = 4,
) -> dict[str, float]:
    """Estimate GPU memory usage in MB.

    Estimates:
        - Model parameters
        - Optimizer states (Adam: 2x parameters for m and v)
        - Gradients
        - Input tensor
        - Rough activation estimate (2-4x input for U-Net-like models)

    Args:
        model: The model to profile.
        input_shape: Input tensor shape.
        dtype_bytes: Bytes per element (4 for float32, 2 for float16).

    Returns:
        Dict with memory estimates in MB.
    """
    n_params = sum(p.numel() for p in model.parameters())
    input_elements = 1
    for dim in input_shape:
        input_elements *= dim

    param_mb = n_params * dtype_bytes / (1024**2)
    optimizer_mb = n_params * dtype_bytes * 2 / (1024**2)  # Adam m + v
    gradient_mb = n_params * dtype_bytes / (1024**2)
    input_mb = input_elements * dtype_bytes / (1024**2)
    activation_mb = input_mb * 3  # Rough estimate for U-Net

    total_mb = param_mb + optimizer_mb + gradient_mb + input_mb + activation_mb

    return {
        "parameters_mb": round(param_mb, 1),
        "optimizer_mb": round(optimizer_mb, 1),
        "gradients_mb": round(gradient_mb, 1),
        "input_mb": round(input_mb, 1),
        "activations_mb": round(activation_mb, 1),
        "total_estimated_mb": round(total_mb, 1),
    }


def measure_inference_time(
    model: nn.Module,
    input_shape: tuple[int, ...] = (1, 4, 128, 128, 128),
    device: str = "cpu",
    warmup_runs: int = 2,
    timed_runs: int = 5,
) -> dict[str, float]:
    """Measure inference time.

    Args:
        model: Model to benchmark.
        input_shape: Input tensor shape.
        device: "cpu" or "cuda".
        warmup_runs: Number of warmup forward passes.
        timed_runs: Number of timed forward passes.

    Returns:
        Dict with mean and std inference time in seconds.
    """
    model = model.to(device)
    model.eval()
    x = torch.randn(*input_shape, device=device)

    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()

    # Timed runs
    times = []
    with torch.no_grad():
        for _ in range(timed_runs):
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(x)
            if device == "cuda":
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)

    return {
        "mean_seconds": round(sum(times) / len(times), 4),
        "std_seconds": round(
            (sum((t - sum(times) / len(times)) ** 2 for t in times) / len(times)) ** 0.5,
            4,
        ),
        "device": device,
        "runs": timed_runs,
    }


def profile_model(
    model: nn.Module,
    name: str,
    input_shape: tuple[int, ...] = (1, 4, 128, 128, 128),
) -> dict:
    """Full model profile: parameters, memory, speed.

    Returns:
        Dict with all profiling results.
    """
    params = count_parameters(model)
    memory = estimate_memory_mb(model, input_shape)

    return {
        "name": name,
        "parameters": params,
        "memory": memory,
    }


def profile_all_models(
    input_shape: tuple[int, ...] = (1, 4, 64, 64, 64),
) -> str:
    """Profile all OncoSeg models and return formatted comparison table.

    Uses 64³ input to avoid OOM on machines without GPU.
    """
    from src.models.baselines.swin_unetr import SwinUNETRBaseline
    from src.models.baselines.unet3d import UNet3D
    from src.models.baselines.unetr import UNETR
    from src.models.oncoseg import OncoSeg

    models = {
        "OncoSeg": OncoSeg(
            in_channels=4,
            num_classes=4,
            embed_dim=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            deep_supervision=True,
        ),
        "OncoSeg (temporal)": OncoSeg(
            in_channels=4,
            num_classes=4,
            embed_dim=48,
            depths=(2, 2, 2, 2),
            num_heads=(3, 6, 12, 24),
            deep_supervision=True,
            temporal=True,
        ),
        "UNet3D": UNet3D(in_channels=4, num_classes=4),
        "SwinUNETR": SwinUNETRBaseline(in_channels=4, num_classes=4),
        "UNETR": UNETR(in_channels=4, num_classes=4, img_size=(64, 64, 64)),
    }

    lines = [
        "Model Profiling Report",
        "=" * 70,
        f"Input shape: {input_shape}",
        "",
        f"{'Model':22s} {'Params':>12s} {'Trainable':>12s} {'Memory (MB)':>12s}",
        "─" * 70,
    ]

    for name, model in models.items():
        profile = profile_model(model, name, input_shape)
        p = profile["parameters"]
        m = profile["memory"]
        lines.append(
            f"{name:22s} {p['total']:>12,} {p['trainable']:>12,} {m['total_estimated_mb']:>12.1f}"
        )
        del model

    return "\n".join(lines)
