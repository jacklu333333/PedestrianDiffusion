from utils.ronin.source.model_temporal import TCNSeqNetwork
from .diffusionSpectrum import diffusionSpectrum
from .common_imports import *
from .utils import *


class RoNINTCNModule(diffusionSpectrum):
    def __init__(self, config):
        super(RoNINTCNModule, self).__init__(config)
        self.loss = nn.MSELoss()
        if hasattr(self, "special_loss"):
            del self.special_loss
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "VAE"):
            del self.VAE
        if hasattr(self, "scheduler"):
            del self.scheduler

        # In original RoNIN, TCN kernels and layers are heavily configured
        # Default layer_channels usually have around 6 layers for a large receptive field
        self.model = TCNSeqNetwork(
            input_channel=6,
            output_channel=3,
            kernel_size=self.config.get("kernel_size", 3),
            layer_channels=self.config.get("layer_channels", [64, 64, 64, 64, 64, 64]),
            dropout=self.config.get("dropout", 0.1),
        )
        self.toXYZ = nn.Sequential()

    def get_XY(self, x, y):
        return x, y[:, :3]

    def forward(self, x):
        # x is [batch, channels, length]
        x = x.transpose(1, 2)  # TCNSeqNetwork expects [batch, length, channels]
        out = self.model(x)  # [batch, length, out_size]
        return out

    def _get_loss(self, y_hat, y, mode="full"):
        """
        Original RoNIN TCN model used 'part' loss (windowed integration based on receptive field) 
        because TCNs cannot remember infinitely backwards like LSTMs. We provide both options.
        """
        loss_mode = self.config.get("tcn_loss_mode", "full")
        
        y_hat_pos = y_hat.cumsum(dim=-1)
        y_pos = y.cumsum(dim=-1)
        
        if loss_mode == "part":
            # the receptive field of this TCN architecture
            history = self.model.get_receptive_field()
            if y_hat_pos.shape[-1] > history:
                y_pos = y_pos[..., history:] - y_pos[..., :-history]
                y_hat_pos = y_hat_pos[..., history:] - y_hat_pos[..., :-history]

        return self.loss(y_hat_pos, y_pos)

    @overrides
    def regular_forward(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L)
        x, y = self.get_XY(x=x, y=y)

        y_hat = self.forward(x)
        assert torch.isfinite(y_hat).all(), "The estimate is not finite"
        y_hat = y_hat.transpose(1, 2)  # transpose back to [batch, out_size, length]

        losses = {
            f"loss_total/{mode}": self._get_loss(y_hat, y),
        }

        metrics = {}
        for key, metric in self.metrics[mode].items():
            if (mode == "train" or mode == "val") and (
                "pearson" in key or "simVector" in key
            ):
                dictionary = metric(y_hat, y)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            elif "naive" in key:
                dictionary = metric(y_hat, y, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                dictionary = metric(y_hat, y)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v

        self.log_dict(
            dictionary=losses,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True if mode == "train" else False,
            sync_dist=True,
        )

        self.log_dict(
            dictionary=metrics,
            sync_dist=True,
            on_step=True if mode == "train" else False,
            on_epoch=True,
        )

        if mode == "train" and self.config["ema"]["enable"] and hasattr(self, "ema"):
            self.log(
                name=f"ema_decay/{mode}",
                value=self.ema.get_decay(self.global_step),
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        # log current learning rate
        if mode == "train":
            self.log(
                name=f"lr/{mode}",
                value=self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )

        return losses[f"loss_total/{mode}"]

    @overrides
    def sequential_forward(self, batch, mode="test"):
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L, do_noise=False)
        x, y = self.get_XY(x=x, y=y)

        y_hat = self.forward(x)
        y_hat = y_hat.transpose(1, 2)  # transpose back to [batch, out_size, length]

        losses = {
            f"loss_total/{mode}": self._get_loss(y_hat, y),
        }

        metrics = {}
        for key, metric in self.metrics[mode].items():
            if "naive" in key:
                dictionary = metric(y_hat, y, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                dictionary = metric(y_hat, y)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v

        self.log_dict(
            dictionary=metrics,
            sync_dist=True if not mode == "peek_testing" else False,
            on_step=False,
            on_epoch=True,
        )

        self.log_dict(
            dictionary=losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True if mode == "test" else False,
            sync_dist=True if not mode == "peek_testing" else False,
        )

        return losses[f"loss_total/{mode}"], {
            "dataX": y_hat,
            "dataY": y,
            "dataL": L,
            "name": batch["name"],
            "index": batch["index"],
            "label": batch["label"],
        }

    @overrides
    def _generate_metrics(self, suffix):
        metrics = {
            f"metric_naive_distance_error_XY/{suffix}": NaiveDistanceError(
                channel_index=[0, 1],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError(
                channel_index=[2],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_X/{suffix}": NaiveDistanceError(
                channel_index=[0],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_Y/{suffix}": NaiveDistanceError(
                channel_index=[1],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
        }
        return metrics

    @overrides
    def configure_optimizers(self):
        return super().configure_optimizers()