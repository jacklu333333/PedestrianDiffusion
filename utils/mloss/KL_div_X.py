from .common_imports import *


class KL_div_X(nn.Module):
    def __init__(self):
        super(KL_div_X, self).__init__()
        self.metric_x = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_x(x[:, 0], y[:, 0])
        return loss
