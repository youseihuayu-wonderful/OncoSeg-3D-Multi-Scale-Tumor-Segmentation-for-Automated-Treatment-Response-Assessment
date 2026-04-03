"""Unit tests for loss functions."""

import pytest
import torch


class TestDiceCELoss:
    """Test DiceCE combined loss."""

    @pytest.fixture
    def loss_fn(self):
        from src.training.losses import DiceCELoss

        return DiceCELoss(dice_weight=0.5, ce_weight=0.5)

    def test_output_is_scalar(self, loss_fn):
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.zeros(2, 4, 16, 16, 16)
        target[:, 0] = 1.0  # All background
        loss = loss_fn(pred, target)
        assert loss.dim() == 0  # Scalar

    def test_loss_is_positive(self, loss_fn):
        pred = torch.randn(2, 4, 16, 16, 16)
        target = torch.zeros(2, 4, 16, 16, 16)
        target[:, 0] = 1.0
        loss = loss_fn(pred, target)
        assert loss.item() > 0

    def test_perfect_prediction_low_loss(self, loss_fn):
        """A near-perfect prediction should have lower loss than random."""
        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0  # All class 0

        # Near-perfect: high logit for class 0
        good_pred = torch.zeros(1, 4, 16, 16, 16)
        good_pred[:, 0] = 10.0

        # Random prediction
        bad_pred = torch.randn(1, 4, 16, 16, 16)

        good_loss = loss_fn(good_pred, target)
        bad_loss = loss_fn(bad_pred, target)

        assert good_loss.item() < bad_loss.item()

    def test_gradient_flows(self, loss_fn):
        pred = torch.randn(1, 4, 16, 16, 16, requires_grad=True)
        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape


class TestDeepSupervisionLoss:
    """Test deep supervision weighted loss."""

    def test_weighted_sum(self):
        from src.training.losses import DeepSupervisionLoss, DiceCELoss

        base_loss = DiceCELoss()
        ds_loss = DeepSupervisionLoss(base_loss)

        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0

        predictions = [
            torch.randn(1, 4, 16, 16, 16),
            torch.randn(1, 4, 16, 16, 16),
            torch.randn(1, 4, 16, 16, 16),
        ]

        loss = ds_loss(predictions, target)
        assert loss.dim() == 0
        assert loss.item() > 0

    def test_single_prediction(self):
        from src.training.losses import DeepSupervisionLoss, DiceCELoss

        base_loss = DiceCELoss()
        ds_loss = DeepSupervisionLoss(base_loss)

        target = torch.zeros(1, 4, 16, 16, 16)
        target[:, 0] = 1.0
        predictions = [torch.randn(1, 4, 16, 16, 16)]

        loss = ds_loss(predictions, target)
        assert loss.dim() == 0
