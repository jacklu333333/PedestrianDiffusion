from .common_imports import *


class CauchyLoss(nn.Module):
    def __init__(self, rescale=0.01):
        super(CauchyLoss, self).__init__()
        self.rescale = rescale

    def forward(self, x, y):
        loss = (x.sum(dim=-1) - y.sum(dim=-1)).square() * self.rescale
        return loss.mean()
