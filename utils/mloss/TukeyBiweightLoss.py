from .common_imports import *


class TukeyBiweightLoss(nn.Module):
    def __init__(self, rescale=0.01):
        super(TukeyBiweightLoss, self).__init__()
        self.rescale = rescale
        raise NotImplementedError

    def forward(self, x, y):
        diff = (x - y).abs() / self.rescale
