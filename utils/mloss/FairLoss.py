from .common_imports import *


class FairLoss(nn.Module):
    def __init__(self, rescale=0.01):
        super(FairLoss, self).__init__()
        self.rescale = rescale

    def forward(self, x, y):
        diff = (x - y).abs() / self.rescale
        index = diff < 1
        loss = torch.zeros_like(diff)
        loss[index] = 1 - diff[index].square()
        loss[~index] = 2 * diff[~index] - 1
        return loss.mean()
