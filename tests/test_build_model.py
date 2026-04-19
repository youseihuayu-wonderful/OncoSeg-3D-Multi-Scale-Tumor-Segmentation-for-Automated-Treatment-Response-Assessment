"""Regression tests for train_all.build_model factory.

Protects the 6-hour Kaggle / Colab runs from crashing on a misconfigured
model after minutes of dataset download — every supported model name
must instantiate and forward-pass a small CPU tensor to (B, 3, *ROI).
"""

import pytest
import torch

from train_all import NUM_CLASSES, build_model

ROI = (64, 64, 64)
BATCH = 1
IN_CHANNELS = 4


def _extract_main_pred(out):
    """OncoSeg with deep supervision returns a dict; baselines return a tensor."""
    if isinstance(out, dict):
        return out["pred"]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


@pytest.mark.parametrize(
    "name",
    [
        "oncoseg",
        "oncoseg_no_xattn",
        "oncoseg_no_ds",
        "oncoseg_no_mcdrop",
        "oncoseg_small",
        "unet3d",
        "swin_unetr",
        "unetr",
    ],
)
def test_build_model_forward_shape(name):
    model = build_model(name, roi_size=ROI)
    model.eval()
    x = torch.randn(BATCH, IN_CHANNELS, *ROI)
    with torch.no_grad():
        out = model(x)
    pred = _extract_main_pred(out)
    assert pred.shape == (BATCH, NUM_CLASSES, *ROI), (
        f"{name} returned {tuple(pred.shape)}, expected {(BATCH, NUM_CLASSES, *ROI)}"
    )


def test_build_model_rejects_unknown_name():
    with pytest.raises(ValueError, match="Unknown"):
        build_model("not_a_real_model", roi_size=ROI)


def test_oncoseg_small_has_fewer_params_than_baseline():
    baseline = build_model("oncoseg", roi_size=ROI)
    small = build_model("oncoseg_small", roi_size=ROI)
    n_base = sum(p.numel() for p in baseline.parameters())
    n_small = sum(p.numel() for p in small.parameters())
    assert n_small < n_base, (
        f"oncoseg_small ({n_small}) should have fewer params than oncoseg ({n_base})"
    )


def test_oncoseg_no_xattn_differs_from_baseline():
    """Cross-attention removal must change parameter count — otherwise the
    ablation knob is a no-op and the paper's ablation table is meaningless."""
    baseline = build_model("oncoseg", roi_size=ROI)
    no_xattn = build_model("oncoseg_no_xattn", roi_size=ROI)
    n_base = sum(p.numel() for p in baseline.parameters())
    n_no_x = sum(p.numel() for p in no_xattn.parameters())
    assert n_no_x != n_base, (
        "oncoseg_no_xattn has same param count as oncoseg — cross-attention "
        "knob is not wired correctly"
    )
