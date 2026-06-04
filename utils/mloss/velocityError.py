from .common_imports import *


class velocityError(nn.Module):
    def __init__(self):
        super(velocityError, self).__init__()
        # self.loss = nn.HuberLoss(delta=1)
        self.loss = nn.L1Loss()
        # self.loss = nn.MSELoss()

    def forward(self, x, y):
        # loss = self.loss(
        #     x.cumsum(dim=-1),  # * 0.01,
        #     y.cumsum(dim=-1),  # * 0.01,
        # )
        loss = x[:, :3].cumsum(dim=-1) - y[:, :3].cumsum(dim=-1)
        loss = loss.norm(dim=1) * 0.01
        loss = F.mse_loss(loss, torch.zeros_like(loss))
        return loss.mean()
