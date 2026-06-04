# utils/mloss_vae.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ControlVAELossTime(nn.Module):
    def __init__(
        self, latent_space_numel, target_kl=0.1, kl_beta=1.0, Kp=0.01, Ki=0.0001
    ):
        super().__init__()
        self.latent_space_numel = latent_space_numel
        self.target_kl = target_kl
        self.register_buffer("beta", torch.tensor(kl_beta))
        self.Kp = Kp
        self.Ki = Ki
        self.register_buffer("error_integral", torch.tensor(0.0))
        self.min_beta = 0.0
        self.max_beta = 1.0
        self.log_vars = nn.Parameter(torch.zeros(2, 1))

    def update_beta(self, kl_div):
        # ControlVAE: PI controller to adjust beta
        # We want KL divergence to be close to target_kl
        # If KL > target, we increase beta to penalize it more.
        error = kl_div - self.target_kl
        self.error_integral = self.error_integral + error
        # Clamp integral to avoid windup if needed, but keeping simple for now

        new_beta = self.beta + self.Kp * error + self.Ki * self.error_integral
        new_beta = torch.clamp(new_beta, self.min_beta, self.max_beta)
        self.beta = new_beta

    # def forward(self, recon_x, x, mu, logvar):
    #     # recon_x, x: (B, C, T)

    #     # Reconstruction loss (MSE)
    #     # Assuming 6 channels: 0-2 acc, 3-5 gyr
    #     recon_loss_acc = F.mse_loss(recon_x[:, :3], x[:, :3])
    #     recon_loss_gyr = F.mse_loss(recon_x[:, 3:], x[:, 3:])
    #     recon_loss = recon_loss_acc + recon_loss_gyr

    #     # KL Divergence
    #     # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
    #     # Sum over latent dimensions, mean over batch
    #     kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    #     kl_loss = torch.mean(kl_loss)

    #     if self.training:
    #         self.update_beta(kl_loss.detach())

    #     total_loss = recon_loss + self.beta * kl_loss

    #     return {
    #         "loss_total": total_loss,
    #         "recon_loss_acc": F.mse_loss(
    #             recon_x[:, :3] * 1000, x[:, :3] * 1000
    #         ),  # Monitor the massive raw value
    #         "recon_loss_gyr": F.mse_loss(
    #             recon_x[:, 3:] * 1800 / torch.pi, x[:, 3:] * 1800 / torch.pi
    #         ),
    #         "kl_loss": kl_loss,
    #         "beta": self.beta,
    #     }
    def forward(self, recon_x, x, mu, logvar):
        # recon_x, x: (B, C, T)

        # Reconstruction loss (MSE)
        # Assuming 6 channels: 0-2 acc, 3-5 gyr
        scale = 1e-3
        recon_loss_acc = F.huber_loss(
            recon_x[:, :3] / scale, x[:, :3] / scale, delta=scale
        )
        recon_loss_gyr = F.huber_loss(
            recon_x[:, 3:] / scale, x[:, 3:] / scale, delta=scale
        )
        # recon_loss = recon_loss_acc + recon_loss_gyr
        recon_loss = 0
        for idx, l in enumerate([recon_loss_acc, recon_loss_gyr]):
            recon_loss += torch.exp(-self.log_vars[idx]) * l + self.log_vars[idx]

        # # KL Divergence
        # # -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        # # Sum over latent dimensions, mean over batch
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        # kl_loss = torch.mean(kl_loss)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss = torch.mean(kl_loss)
        kl_loss = kl_loss / self.latent_space_numel
        # masking
        if self.training:
            self.update_beta(kl_loss.detach())

        kl_loss_mask = torch.min(kl_loss, torch.tensor(0.1, device=kl_loss.device))

        total_loss = recon_loss + kl_loss_mask

        return {
            "loss_total": total_loss,
            "recon_loss_acc": F.mse_loss(
                recon_x[:, :3] * 1000, x[:, :3] * 1000
            ),  # Monitor the massive raw value
            "recon_loss_gyr": F.mse_loss(
                recon_x[:, 3:] * 1800 / torch.pi, x[:, 3:] * 1800 / torch.pi
            ),
            "kl_loss": kl_loss,
            "beta": self.beta,
        }
