from .common_imports import *


class ControlVAELoss(nn.Module):
    def __init__(
        self,
        latent_space_numel,
        target_kl=10.0,
        kl_kp=0.01,
        kl_ki=0.0001,
        beta_min=1e-5,
        beta_max=1000.0,
    ):
        super().__init__()
        # 1. Reconstruction Weights (Observation Noise)
        self.log_vars = nn.Parameter(torch.zeros(4))

        self.latent_space_numel = latent_space_numel

        # 2. ControlVAE Parameters
        # target_kl: The desired capacity (nats) per batch item.
        # Start with ~5.0 - 15.0 for sensor data.
        self.target_kl = target_kl
        self.kp = kl_kp  # Proportional gain
        self.ki = kl_ki  # Integral gain
        self.beta_min = beta_min
        self.beta_max = beta_max

        # We register beta as a buffer so it's saved with state_dict
        # but not updated by the optimizer directly.
        self.register_buffer("kl_beta", torch.tensor(1.0))
        self.register_buffer("kl_error_integral", torch.tensor(0.0))

    def update_beta(self, current_kl):
        # --- THE FIX ---
        # Logic: If Current KL (200) > Target (64), we want Error to be POSITIVE (+136)
        # This increases Beta, which creates a stronger penalty.
        error = current_kl.item() - self.target_kl

        self.kl_error_integral += error

        # Calculate scale factor
        # We use a multiplicative update: Beta_new = Beta_old * exp(PID_output)
        pid_output = self.kp * error + self.ki * self.kl_error_integral

        # Clamp the PID output to avoid massive jumps in a single step (e.g., don't multiply by 1000x)
        # Limiting the exponent to +/- 2.0 keeps the update smooth.
        pid_output = torch.clamp(torch.tensor(pid_output), min=-2.0, max=2.0)

        new_beta = self.kl_beta * torch.exp(pid_output)

        # Clamp final beta
        self.kl_beta = torch.clamp(new_beta, min=self.beta_min, max=self.beta_max)

    def forward(self, recon_x_f, x_f, recon_x_t, x_t, mu, logvar):
        # --- PART A: Reconstruction (Unchanged) ---
        cof = 1e-4
        recon_t_loss_acc = F.mse_loss(recon_x_t[:, :3] / cof, x_t[:, :3] / cof)

        cof = 1e-2 / 180 * torch.pi
        recon_t_loss_gyr = F.mse_loss(recon_x_t[:, 3:] / cof, x_t[:, 3:] / cof)

        recon_f_loss_acc = F.huber_loss(
            recon_x_f[:, :3] * 1e3, x_f[:, :3] * 1e3, delta=1e-3
        )
        recon_f_loss_gyr = F.huber_loss(
            recon_x_f[:, 3:] * 1e3, x_f[:, 3:] * 1e3, delta=1e-3
        )

        losses = [
            recon_t_loss_acc,
            recon_t_loss_gyr,
            recon_f_loss_acc,
            recon_f_loss_gyr,
        ]

        # Weighted Reconstruction Loss (Kendall's approach)
        recon_loss_weighted = 0
        for i, l in enumerate(losses):
            recon_loss_weighted += torch.exp(-self.log_vars[i]) * l + self.log_vars[i]

        # --- PART B: The KL Regularization (Dynamic Beta) ---

        # Calculate raw KL
        # Note: Sum over dims, mean over batch is standard
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_loss_mean = torch.mean(kl_div)  # Average KL per sample
        kl_loss_mean = kl_loss_mean / self.latent_space_numel

        # --- AUTOMATIC TUNING ---
        if self.training:
            # Update beta based on how far we are from target KL
            self.update_beta(kl_loss_mean.detach())

        # Apply the dynamic beta
        weighted_kl_loss = self.kl_beta * kl_loss_mean

        # --- Total ---
        total_loss = recon_loss_weighted + weighted_kl_loss

        return {
            "loss_total": total_loss,
            "kl_loss": kl_loss_mean,
            "current_beta": self.kl_beta,  # Monitor this!
            "recon_loss_acc": F.mse_loss(
                recon_x_t[:, :3] * 1000, x_t[:, :3] * 1000
            ),  # Monitor the massive raw value
            "recon_loss_gyr": F.mse_loss(
                recon_x_t[:, 3:] * 1800 / torch.pi, x_t[:, 3:] * 1800 / torch.pi
            ),
        }
