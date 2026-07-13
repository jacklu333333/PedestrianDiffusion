from utils.ronin.source.model_temporal import TCNSeqNetwork
from .diffusionSpectrum import diffusionSpectrum
from .common_imports import *
from .utils import *
from ..mmodels.DeepILS import FCOutputModule, BasicBlock1D


class DeepILSModule(diffusionSpectrum):
    def __init__(self, config):
        super(DeepILSModule, self).__init__(config)
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
        _fc_config = {'fc_dim': 512, 'in_dim': 7, 'dropout': 0.5, 'trans_planes': 128}
        n_class = 2
        self.model = DeepILS(6, n_class, BasicBlock1D, [2, 2, 2, 2],
                           base_plane=64, output_block=FCOutputModule, kernel_size=3, **_fc_config)
        self.toXYZ = nn.Sequential()
        torch.use_deterministic_algorithms(False)

    def get_XY(self, x, y):
        return x, y[:, :2].sum(dim=-1)

    def forward(self, x):
        # x is [batch, channels, length]
        out = self.model(x)  # [batch, length, out_size]
        return out

    def _get_loss(self, y_hat, y, mode="full"):
        """
        Original RoNIN TCN model used 'part' loss (windowed integration based on receptive field) 
        because TCNs cannot remember infinitely backwards like LSTMs. We provide both options.
        """
        
        y_hat_pos = y_hat
        y_pos = y
    
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
        y_hat = y_hat

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
        y_hat = torch.cat([y_hat, torch.zeros_like(y_hat)[:,:1]],dim=1)
        y = torch.cat([y, torch.zeros_like(y)[:,:1]],dim=1)
        return losses[f"loss_total/{mode}"], {
            "dataX": y_hat.reshape(-1,3,1),
            "dataY": y.reshape(-1,3,1),
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
            # f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError(
            #     channel_index=[2],
            #     sampling_rate=self.config["sampling_rate"],
            #     avg_win=self.config["target_duration"],
            # ),
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