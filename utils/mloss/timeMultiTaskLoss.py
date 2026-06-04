from .common_imports import *
from .simclr_loss import simclr_loss


class timeMultiTaskLoss(nn.Module):
    def __init__(self, dt, channel_dict=None, sim_priority=5.0):
        super(timeMultiTaskLoss, self).__init__()

        # 1. Add a priority scalar (Strategy 1)
        # Values > 1.0 will force the optimizer to prioritize this loss
        self.sim_priority = sim_priority

        if channel_dict is None:
            channel_dict = {"acc": 3, "gyr": 3}
        self.channel_dict = channel_dict
        self.nb_outputs = len(channel_dict)
        self.dt = dt

        # learnable log variances: 3 objectives (Time, Sim, Sum)
        self.log_vars = nn.Parameter(torch.zeros(self.nb_outputs, 6))

        # # 2. Biased Initialization (Strategy 2)
        # # Initialize the 'Sim' column (index 1) to -1.5.
        # # exp(-(-1.5)) approx 4.5. This starts training with 4.5x weight on Sim.
        # # We keep others at 0 (1x weight).
        # with torch.no_grad():
        #     self.log_vars[:, 1].fill_(-1.5)

        self.sim_loss = simclr_loss()

    def forward(self, y_hat_t, y_t, timesteps):
        total_loss = 0
        losses = {}
        y_hat_t = y_hat_t.double()
        y_t = y_t.double()

        # check shapes
        assert y_hat_t.shape == y_t.shape, f"{y_hat_t.shape} vs {y_t.shape}"
        assert y_hat_t.shape[1] == sum(
            self.channel_dict.values()
        ), f"{y_hat_t.shape[1]} vs {sum(self.channel_dict.values())}"

        start_idx = 0
        for i, (name, ch) in enumerate(self.channel_dict.items()):
            end_idx = start_idx + ch

            # --- Objective 1: General MSE (Time) ---
            time_loss_val = (
                (y_hat_t[:, start_idx:end_idx] - y_t[:, start_idx:end_idx])
                .norm(p=2, dim=1)
                .mean()
            )

            norm_loss = F.huber_loss(
                time_loss_val * 1e3,
                torch.zeros_like(time_loss_val),
                delta=1e-3,
            )

            position_loss = F.huber_loss(
                y_hat_t[:, start_idx:end_idx].cumsum(dim=-1) * self.dt * 1e3,
                y_t[:, start_idx:end_idx].cumsum(dim=-1) * self.dt * 1e3,
                delta=1e-3,
            )

            vel_loss = F.huber_loss(
                y_hat_t[:, start_idx:end_idx] * 1e3,
                y_t[:, start_idx:end_idx] * 1e3,
                delta=1e-3,
            )

            acc_loss = F.huber_loss(
                y_hat_t[:, start_idx:end_idx].diff(dim=-1) / self.dt * 1e3,
                y_t[:, start_idx:end_idx].diff(dim=-1) / self.dt * 1e3,
                delta=1e-3,
            )

            # --- Objective 2: Cosine Similarity (Sim) ---
            sim_loss_val = self.sim_loss(
                y_hat_t[:, start_idx:end_idx],
                y_t[:, start_idx:end_idx],
            )
            sim_loss = F.huber_loss(
                sim_loss_val * 1e3, torch.zeros_like(sim_loss_val), delta=1e-3
            )

            # # Calculate standard uncertainty loss
            # uncertainty_sim = (
            #     torch.exp(-self.log_vars[i, 1]) * sim_loss_val + self.log_vars[i, 1]
            # )

            # # Apply Static Priority Multiplier
            # # This prevents the model from "optimizing away" the loss by just increasing log_var
            # sim_loss = self.sim_priority * uncertainty_sim

            # --- Objective 3: Sum Loss ---
            # cof = 1e-4
            y_sum = y_t[:, start_idx:end_idx].sum(dim=-1) * self.dt
            y_hat_sum = y_hat_t[:, start_idx:end_idx].sum(dim=-1) * self.dt
            sum_loss_val = (y_sum - y_hat_sum).norm(p=2, dim=1).mean()
            sum_loss = F.huber_loss(y_hat_sum * 1e3, y_sum * 1e3, delta=1e-3)
            # target_energy = y_t.norm(p=2, dim=1, keepdim=True)  # (B, 1, T)
            # is_stationary = (target_energy < 0.1).float()
            # loss_zvup = (y_hat_t.abs() * is_stationary).mean()

            # # Combine all
            loss_list = [
                sim_loss,
                norm_loss,
                position_loss,
                vel_loss,
                acc_loss,
                sum_loss,
                # loss_zvup,
            ]
            # total_loss += time_loss + sim_loss + sum_loss

            for idxl, l in enumerate(loss_list):
                total_loss += (
                    torch.exp(-self.log_vars[i, idxl]) * l + self.log_vars[i, idxl]
                )

            # Logging for debug
            losses[f"loss_TimeSUM_{name}"] = sum_loss_val
            losses[f"loss_TimeSIM_{name}"] = sim_loss_val
            losses[f"loss_Time_{name}"] = time_loss_val
            losses[f"loss_Norm_{name}"] = norm_loss

            start_idx = end_idx

        losses["loss_total"] = total_loss
        return losses
