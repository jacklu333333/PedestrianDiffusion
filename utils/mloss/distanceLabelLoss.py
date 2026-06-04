from .common_imports import *


class distanceLabelLoss(nn.Module):
    def __init__(self):
        super(distanceLabelLoss, self).__init__()
        self.loss_distance = nn.L1Loss()
        # self.loss_distance = nn.MSELoss()
        # self.loss_class = nn.CrossEntropyLoss()

    def forward(self, x, y):
        loss_distance = self.loss_distance(x, y.float())
        # loss_class = self.loss_class(x, y)

        # return loss_distance + loss_class
        return loss_distance
