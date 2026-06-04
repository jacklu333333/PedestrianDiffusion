from .common_imports import *

# def ssim_1d(x, y, window_size=11, C1=1e-7, C2=1e-7):
#     """
#     x, y: (B, C, L)
#     Joint-channel SSIM over the whole multivariate signal.
#     """
#     B, C, L = x.shape

#     # Use a 1D Gaussian window for local statistics
#     def gaussian_window(window_size, sigma):
#         gauss = (
#             torch.arange(window_size, dtype=x.dtype, device=x.device)
#             - (window_size - 1) / 2
#         )
#         gauss = torch.exp(-0.5 * (gauss / sigma) ** 2)
#         gauss = gauss / gauss.sum()
#         return gauss.view(1, 1, window_size)

#     sigma = window_size / 6.0  # Standard choice: covers ~99% of mass
#     window = gaussian_window(window_size, sigma)

#     # first compute local temporal mean per channel
#     mu_x = F.conv1d(x, window.expand(C, 1, window_size), padding="same", groups=C)
#     mu_y = F.conv1d(y, window.expand(C, 1, window_size), padding="same", groups=C)

#     # then average over channels -> joint mean
#     mu_x = mu_x.mean(dim=1, keepdim=True)
#     mu_y = mu_y.mean(dim=1, keepdim=True)

#     mu_x2 = mu_x.pow(2)
#     mu_y2 = mu_y.pow(2)
#     mu_xy = mu_x * mu_y

#     sigma_x2 = ((x - mu_x) ** 2).mean(dim=1, keepdim=True)
#     sigma_y2 = ((y - mu_y) ** 2).mean(dim=1, keepdim=True)
#     sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=1, keepdim=True)

#     ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
#         (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
#     )

#     return ssim_map


class metricsSSIM(nn.Module):
    def __init__(
        self,
        data_range=1.0,
        window_size=11,
        window_type="gaussian",
        channel_index=[0, 1, 2],
        reduction="mean",
    ):
        super().__init__()
        self.reduction = reduction
        self.data_range = data_range
        self.window_size = window_size
        # self.sigma = window_size / 6.0  # Standard choice: covers ~99% of mass
        self.sigma = int(self.window_size * 0.99)
        self.window = None  # Will be initialized on first forward
        assert window_type in (
            "gaussian",
            "uniform",
        ), "window_type must be 'gaussian' or 'uniform'"
        self.window_type = window_type
        self.channel_index = channel_index

    def _get_window(self, device, dtype, C):
        if (
            self.window is not None
            and self.window.device == device
            and self.window.dtype == dtype
            and self.window.shape[0] == 1
            and self.window.shape[1] == 1
        ):
            return self.window
        if self.window_type == "gaussian":
            gauss = (
                torch.arange(self.window_size, dtype=dtype, device=device)
                - (self.window_size - 1) / 2
            )
            gauss = torch.exp(-0.5 * (gauss / self.sigma) ** 2)
            gauss = gauss / gauss.sum()
            window = gauss.view(1, 1, self.window_size)
        elif self.window_type == "uniform":
            window = torch.ones(1, 1, self.window_size, dtype=dtype, device=device)
            window = window / self.window_size
        self.window = window
        return window

    def ssim_1d(self, x, y, C1=1e-7, C2=1e-7):
        # `preds`/`target` are already channel-sliced by `forward()` when
        # `self.channel_index` is set. Avoid re-indexing here which can
        # introduce an extra dimension (e.g. when `channel_index` is None)
        # and cause unpacking errors. Expect `x, y` shaped (B, C, L).
        B, C, L = x.shape
        window = self._get_window(x.device, x.dtype, C)
        kernel = window.expand(C, 1, self.window_size)
        # Compute explicit padding to avoid PyTorch 'same' padding warning
        dilation = 1
        pad_total = dilation * (self.window_size - 1)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        if pad_left == pad_right:
            mu_x = F.conv1d(x, kernel, padding=pad_left, groups=C)
            mu_y = F.conv1d(y, kernel, padding=pad_left, groups=C)
        else:
            x_padded = F.pad(x, (pad_left, pad_right))
            y_padded = F.pad(y, (pad_left, pad_right))
            mu_x = F.conv1d(x_padded, kernel, padding=0, groups=C)
            mu_y = F.conv1d(y_padded, kernel, padding=0, groups=C)
        mu_x = mu_x.mean(dim=1, keepdim=True)
        mu_y = mu_y.mean(dim=1, keepdim=True)
        mu_x2 = mu_x.pow(2)
        mu_y2 = mu_y.pow(2)
        mu_xy = mu_x * mu_y
        sigma_x2 = ((x - mu_x) ** 2).mean(dim=1, keepdim=True)
        sigma_y2 = ((y - mu_y) ** 2).mean(dim=1, keepdim=True)
        sigma_xy = ((x - mu_x) * (y - mu_y)).mean(dim=1, keepdim=True)
        ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
            (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
        )
        return ssim_map

    def forward(self, preds, target):
        # preds, target: (B, C, ...)
        # Optionally select subset of channels before computing SSIM
        if self.channel_index is not None:
            preds = preds[:, self.channel_index]
            target = target[:, self.channel_index]
        # Compute SSIM over the selected channels as a whole
        ssim_vals = self.ssim_1d(preds, target)  # (B,) or scalar
        mean = ssim_vals.mean()
        std = ssim_vals.std()
        result = {
            "mean": mean,
            "std": std,
        }
        if self.reduction is not "mean":
            result["loss"] = 1 - ssim_vals  # Convert SSIM to loss (1 - SSIM)
        return result


class LossSSIM(metricsSSIM):
    def __init__(
        self,
        data_range=1.0,
        window_size=11,
        window_type="gaussian",
        channel_index=None,
        reduction="mean",
    ):
        super().__init__(
            data_range=data_range,
            window_size=window_size,
            window_type=window_type,
            channel_index=channel_index,
            reduction="none",
        )
        self.reduc = reduction

    def forward(self, preds, target):
        result = super().forward(preds, target)
        loss = result["loss"]
        if self.reduc == "mean":
            loss = loss.mean()
        elif self.reduc == "sum":
            loss = loss.sum()
        elif self.reduc == "none":
            loss = loss
        return loss
