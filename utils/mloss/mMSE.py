from .common_imports import *


class mMSE(nn.Module):
    def __init__(self, channel_index: list, norm=False):
        super(mMSE, self).__init__()
        self.loss = nn.MSELoss()
        self.channel_index = channel_index
        self.norm = norm

    def forward(self, x, y):
        x = x[:, self.channel_index]
        y = y[:, self.channel_index]

        if self.norm:
            x = x.norm(dim=1)
            y = y.norm(dim=1)

        loss = self.loss(x, y)
        return loss
