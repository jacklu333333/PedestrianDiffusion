from .common_imports import *


class KL_div(nn.Module):
    def __init__(self):
        super(KL_div, self).__init__()
        self.kl_div_x = KL_div_X()
        self.kl_div_y = KL_div_Y()
        self.kl_div_z = KL_div_Z()

    def forward(self, x, y):
        loss_x = self.kl_div_x(x, y)
        loss_y = self.kl_div_y(x, y)
        loss_z = self.kl_div_z(x, y)
        loss = loss_x + loss_y + loss_z
        return loss
