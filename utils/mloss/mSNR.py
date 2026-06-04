from .common_imports import *


class mSNR(nn.Module):
    def __init__(self, channel_index: list, norm=False):
        super(mSNR, self).__init__()
        self.loss = torch.vmap(torch.vmap(signal_noise_ratio))
        self.channel_index = channel_index
        self.norm = norm

    def forward(self, x, y):
        x = x[:, self.channel_index]
        y = y[:, self.channel_index]

        if self.norm:
            x = x.norm(dim=1, keepdim=True)
            y = y.norm(dim=1, keepdim=True)

        loss = self.loss(x, y).mean(dim=-1)
        return {
            "mean": loss.mean(),
            "std": loss.std(),
        }
