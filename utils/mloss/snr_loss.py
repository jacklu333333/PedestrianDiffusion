import torch
import torch.nn as nn
import torch.nn.functional as F


class SNRLoss(nn.Module):
    """
    Signal-to-Noise Ratio (SNR) weighted loss function.

    This loss function computes the Mean Squared Error (MSE) between predictions
    and targets, and then weights each sample's loss by a function of its
    corresponding Signal-to-Noise Ratio (SNR). This is a common technique used
    in training diffusion models.

    The loss for a single sample is calculated as: weight * (preds - target)^2
    The default weighting scheme is 1 / SNR.
    """

    def __init__(self, reduction="mean"):
        """
        Initializes the SNRLoss module.

        Args:
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'.
        """
        super().__init__()
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(
                f"Invalid reduction type: {reduction}. "
                "Supported types are 'mean', 'sum', 'none'."
            )
        self.reduction = reduction

    def forward(
        self, preds: torch.Tensor, target: torch.Tensor, snr: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes the forward pass of the SNR-weighted loss.

        Args:
            preds (torch.Tensor): The predictions from the model.
                                  Shape: (batch_size, ...).
            target (torch.Tensor): The ground truth values.
                                   Shape: (batch_size, ...), same as preds.
            snr (torch.Tensor): The signal-to-noise ratio for each item in the batch.
                                Shape: (batch_size,).

        Returns:
            torch.Tensor: The computed loss. A scalar if reduction is 'mean' or 'sum'.
        """
        # Calculate the squared error
        error = F.mse_loss(preds, target, reduction="none")

        # Reshape error and snr for broadcasting
        # error shape: (batch_size, C, H, W) -> (batch_size, -1)
        # snr shape: (batch_size,) -> (batch_size, 1)
        error = error.flatten(start_dim=1)
        snr = snr.view(-1, 1)

        # Clamp SNR to avoid division by zero or very small numbers
        snr = torch.clamp(snr, min=1e-8)

        # Define the loss weight, a common choice is 1/SNR
        loss_weight = 1.0 / snr

        # Apply the weight to the error
        loss = loss_weight * error

        # Apply reduction
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # 'none'
            return loss
