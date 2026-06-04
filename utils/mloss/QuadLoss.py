from .common_imports import *


class QuadLoss(nn.Module):
    def __init__(self, reduction="mean"):
        super(QuadLoss, self).__init__()
        self.reduction = reduction

    def forward(self, x, y):
        loss = (x - y).square().square()
        if self.reduction == "sum":
            return loss.sum()
        elif self.reduction == "mean":
            return loss.mean()

        return loss
