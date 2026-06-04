from .common_imports import *


class angularError(nn.Module):
    def __init__(self):
        super(angularError, self).__init__()
        # self.loss = nn.HuberLoss(delta=0.0001)
        self.loss = nn.L1Loss()
        # self.loss = nn.MSELoss()

    def forward(self, x, y):
        # loss = (
        #     x.cumsum(dim=-1).cumsum(dim=-1) * 0.01**2
        #     - y.cumsum(dim=-1).cumsum(dim=-1) * 0.01**2
        # ).norm(dim=1)
        loss = self.loss(
            x[:, 3:].cumsum(dim=-1) * 0.01,
            y[:, 3:].cumsum(dim=-1) * 0.01,
        )
        return loss.mean()
