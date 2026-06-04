from .common_imports import *


class angularVelError(nn.Module):
    def __init__(self):
        super(angularVelError, self).__init__()
        self.loss = nn.L1Loss()
        # self.loss = nn.MSELoss()

    def forward(self, x, y):
        # loss = self.loss(
        #     x.cumsum(dim=-1),  # * 0.01,
        #     y.cumsum(dim=-1),  # * 0.01,
        # )
        assert x.shape[1] == 6, f"{x.shape}"
        assert y.shape[1] == 6, f"{y.shape}"

        loss = x[:, 3:] - y[:, 3:]
        loss = loss.norm(dim=1)
        loss = F.mse_loss(loss, torch.zeros_like(loss))
        return loss.mean()
