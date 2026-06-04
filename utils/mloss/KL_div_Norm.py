from .common_imports import *


class KL_div_Norm(nn.Module):
    def __init__(self):
        super(KL_div_Norm, self).__init__()
        self.metric_norm = KLDivergence(log_prob=True)

    def forward(self, x, y):
        loss = self.metric_norm(x.norm(dim=1), y.norm(dim=1))
        return loss
