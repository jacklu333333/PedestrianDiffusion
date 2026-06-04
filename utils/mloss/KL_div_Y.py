from .common_imports import *


class KL_div_Y(nn.Module):
    def __init__(self):
        super(KL_div_Y, self).__init__()
        self.metric_y = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_y(x[:, 1], y[:, 1])
        return loss
