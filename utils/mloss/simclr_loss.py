from .common_imports import *
from .CosSimMetric import CosSimMetric


class simclr_loss(nn.Module):
    def __init__(self, reduction="mean"):
        super(simclr_loss, self).__init__()
        self.metric_x = CosSimMetric(
            channel_index=[0, 1, 2], norm=False, reduction="none"
        )
        # self.metric_y = CosSimMetric(channel_index=[1], norm=False)
        # self.metric_z = CosSimMetric(channel_index=[2], norm=False)
        self.reduction = reduction

    def forward(self, x, y):
        # loss_x = 1 - self.metric_x(x, y)["mean"]
        # loss_y = 1 - self.metric_y(x, y)["mean"]
        # loss_z = 1 - self.metric_z(x, y)["mean"]
        # loss = (loss_x + loss_y + loss_z) / 3
        loss = 1 - self.metric_x(x, y)["loss"]
        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()
        elif self.reduction == "none":
            loss = loss
        return loss
