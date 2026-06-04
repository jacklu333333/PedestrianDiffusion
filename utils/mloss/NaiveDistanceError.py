from .common_imports import *


class NaiveDistanceError(nn.Module):
    def __init__(self, channel_index: list, sampling_rate=100, avg_win=60):
        """ """
        super(NaiveDistanceError, self).__init__()
        self.channel_index = channel_index
        self.sampling_rate = sampling_rate
        self.avg_win = avg_win

    def forward(self, x, y, dataL):
        x = x.float()
        y = y.float()
        window_time = dataL.view(-1, 1) / self.sampling_rate
        loss = (
            (
                (
                    x[:, self.channel_index].sum(dim=-1)
                    - y[:, self.channel_index].sum(dim=-1)
                )
            ).norm(dim=-1)
            / self.sampling_rate
            / window_time
        )
        # the secdondary /self.sampling_rate is for integral

        loss = loss * self.avg_win

        return {
            "mean": loss.mean(),
            "std": loss.std(),
        }
