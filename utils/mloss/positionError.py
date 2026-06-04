from .common_imports import *


class positionError(nn.Module):
    def __init__(self):
        super(positionError, self).__init__()
        # self.loss = nn.HuberLoss(delta=0.0001)
        self.loss = nn.L1Loss()
        # self.loss = nn.MSELoss()

    def forward(self, x, y):
        # loss = (
        #     x.cumsum(dim=-1).cumsum(dim=-1) * 0.01**2
        #     - y.cumsum(dim=-1).cumsum(dim=-1) * 0.01**2
        # ).norm(dim=1)
        loss = self.loss(
            x[:, :3].cumsum(dim=-1).cumsum(dim=-1) * 0.01**2,
            y[:, :3].cumsum(dim=-1).cumsum(dim=-1) * 0.01**2,
        )
        return loss.mean()
