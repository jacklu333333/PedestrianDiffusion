from .common_imports import *


class KL_div_Z(nn.Module):
    def __init__(self):
        super(KL_div_Z, self).__init__()
        self.metric_z = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_z(x[:, 2], y[:, 2])
        return loss
