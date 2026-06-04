from .common_imports import *


class VAELoss(nn.Module):
    def __init__(self, latent_space_numel, kl_threshold=0.1, kl_beta=1.0):
        super().__init__()
        # 1. KEEP: Learnable weights, but ONLY for the 2 sensors (Observation Noise)
        # We removed the 3rd weight because KL is not an observation.
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.latent_space_numel = latent_space_numel
        self.kl_threshold = kl_threshold

        # 2. NEW: A distinct weight for the KL term
        # Since log_vars will shrink the recon loss, this can be a normal number (0.1 - 5.0)
        self.kl_beta = kl_beta

    def forward(self, recon_x_f, x_f, recon_x_t, x_t, mu, logvar):
        # recon_loss_acc = F.huber_loss(recon_x[:, :3], x[:, :3], delta=1e-4)
        # recon_loss_gyr = F.huber_loss(
        #     recon_x[:, 3:],
        #     x[:, 3:],
        #     delta=1e-2 / 180 * torch.pi,
        # )
        cof = 1e-4
        # cof = cof**0.5
        recon_t_loss_acc = F.mse_loss(
            recon_x_t[:, :3] / cof,
            x_t[:, :3] / cof,
            # delta=cof,
        )
        cof = 1e-2 / 180 * torch.pi
        # cof = cof**0.5
        recon_t_loss_gyr = F.mse_loss(
            recon_x_t[:, 3:] / cof,
            x_t[:, 3:] / cof,
            # delta=cof,
        )

        recon_f_loss_acc = F.huber_loss(
            recon_x_f[:, :3] * 1e3, x_f[:, :3] * 1e3, delta=1e-3
        )
        recon_f_loss_gyr = F.huber_loss(
            recon_x_f[:, 3:] * 1e3, x_f[:, 3:] * 1e3, delta=1e-3
        )

        # --- PART B: The KL Regularization ---

        # Standard KL
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss = kl_loss / self.latent_space_numel / x_t.shape[0]

        # Free Bits (Hinge)
        kl_loss_clamped = torch.max(
            kl_loss, torch.tensor(self.kl_threshold, device=kl_loss.device)
        )

        # 3. Hybrid Weighting
        # Because Part A output is now small (~14.0), we can use a standard beta weight.
        # We do NOT use log_vars here.
        weighted_kl_loss = self.kl_beta * kl_loss_clamped

        # --- Total ---
        losses = [
            recon_t_loss_acc,
            recon_t_loss_gyr,
            recon_f_loss_acc,
            recon_f_loss_gyr,
        ]
        total_loss = weighted_kl_loss
        for i, l in enumerate(losses):
            total_loss += torch.exp(-self.log_vars[i]) * l + self.log_vars[i]

        return {
            "loss_total": total_loss,
            "recon_loss_acc": F.mse_loss(
                recon_x_t[:, :3] * 1000, x_t[:, :3] * 1000
            ),  # Monitor the massive raw value
            "recon_loss_gyr": F.mse_loss(
                recon_x_t[:, 3:] * 1800 / torch.pi, x_t[:, 3:] * 1800 / torch.pi
            ),
            "kl_loss": kl_loss,
        }
