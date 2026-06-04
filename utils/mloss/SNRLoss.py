from .common_imports import *


class SNRLoss(nn.Module):
    def __init__(
        self,
    ):
        super(SNRLoss, self).__init__()
        self.snr = torch.vmap(torch.vmap(signal_noise_ratio))

    def forward(self, x, y):
        snr = self.snr(x, y).mean()
        snr = -snr + 85

        return snr
