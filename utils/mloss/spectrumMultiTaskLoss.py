from .common_imports import *
from .simclr_loss import simclr_loss


class spectrumMultiTaskLoss(nn.Module):
    def __init__(self, dt, nb_outputs=2):
        super(spectrumMultiTaskLoss, self).__init__()
        self.dt = dt
        self.nb_outputs = nb_outputs
        # learnable log variances for weighting each objective
        self.log_vars = nn.Parameter(torch.zeros(nb_outputs, 5))
        self.sim_loss = simclr_loss()

    def forward(self, y_hat_f, y_f, y_hat_t, y_t, timesteps):
        total_loss = 0
        losses = {}
        # y_hat_t = y_hat_t.double()
        # y_t = y_t.double()
        # y_hat_f = y_hat_f.double()
        # y_f = y_f.double()

        scale = 1e-3
        for i, name in enumerate(["acc", "gyr"]):
            # cof = 1.0 if name == "acc" else (torch.pi / 180)
            cof = 1.0
            # --- Goal 1: General MSE in Freq domain---
            if y_hat_f is not None:
                loss_Freq = F.huber_loss(
                    y_hat_f[:, i * 3 : i * 3 + 3] / scale,
                    y_f[:, i * 3 : i * 3 + 3] / scale,
                    delta=scale,
                )
                loss1 = loss_Freq

            else:
                loss_Freq = torch.tensor(0.0, device=y_hat_t.device)
                loss1 = torch.tensor(0.0, device=y_hat_t.device)

            # --- Goal 2: Special case  in time domain---
            y_sum = y_t[:, i * 3 : i * 3 + 3].sum(dim=-1) * self.dt
            y_hat_sum = y_hat_t[:, i * 3 : i * 3 + 3].sum(dim=-1) * self.dt
            loss_TimeSUM = (y_sum - y_hat_sum).norm(p=2, dim=1).mean()
            loss2 = F.huber_loss(
                y_hat_sum / (scale * cof), y_sum / (scale * cof), delta=scale * cof
            )

            # --- Goal 3: Another general MSE (same as Goal 1) in Time domain ---
            loss_Time = (
                (y_hat_t[:, i * 3 : i * 3 + 3] - y_t[:, i * 3 : i * 3 + 3])
                .norm(p=2, dim=1)
                .mean()
            )
            loss3 = F.huber_loss(
                y_hat_t[:, i * 3 : i * 3 + 3] / (scale * cof) * 10,
                y_t[:, i * 3 : i * 3 + 3] / (scale * cof) * 10,
                delta=scale * cof,
            )

            # --- Goal 4: Similarity loss in time domain ---
            loss_TimeSIM = self.sim_loss(
                y_hat_t[:, i * 3 : i * 3 + 3], y_t[:, i * 3 : i * 3 + 3]
            )
            loss4 = F.huber_loss(
                loss_TimeSIM / scale * 10, torch.zeros_like(loss_TimeSIM), delta=scale
            )

            # --- Goal 5: Similarity loss in freq domain ---
            if y_hat_f is not None:
                loss_FreqSIM = self.sim_loss(
                    y_hat_f[:, i * 3 : i * 3 + 3].reshape(y_hat_f.shape[0], 3, -1),
                    y_f[:, i * 3 : i * 3 + 3].reshape(y_f.shape[0], 3, -1),
                )
                loss5 = F.huber_loss(
                    loss_FreqSIM / scale * 10,
                    torch.zeros_like(loss_FreqSIM),
                    delta=scale,
                )
            else:
                loss_FreqSIM = torch.tensor(0.0, device=y_hat_t.device)
                loss5 = torch.tensor(0.0, device=y_hat_t.device)

            loss_list = [loss1, loss2, loss3, loss4, loss5]
            for idxl, l in enumerate(loss_list):
                total_loss += (
                    torch.exp(-self.log_vars[i, idxl]) * l + self.log_vars[i, idxl]
                )
            # Combine all
            losses[f"loss_Freq_{name}"] = loss_Freq
            losses[f"loss_Time_{name}"] = loss_Time
            if name == "gyr":
                loss_TimeSUM *= 180 / torch.pi  # convert to degrees
            losses[f"loss_TimeSUM_{name}"] = loss_TimeSUM
            losses[f"loss_TimeSIM_{name}"] = loss_TimeSIM
            losses[f"loss_FreqSIM_{name}"] = loss_FreqSIM
        losses["loss_total"] = total_loss
        return losses
