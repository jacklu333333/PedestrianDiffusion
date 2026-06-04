from .common_imports import *


class accelerationError(nn.Module):
    def __init__(self):
        super(accelerationError, self).__init__()
        self.loss = nn.L1Loss()
        # self.loss = nn.MSELoss()

    def forward(self, x, y):
        # loss = (x[:, 2] - y[:, 2]).square().sum(dim=-1).sqrt() * self.rescale
        loss = self.loss(x[:, :3], y[:, :3])
        return loss.mean()
