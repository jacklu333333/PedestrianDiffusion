from utils.TLIO.src.network.losses import get_loss
from utils.TLIO.src.network.model_factory import get_model

from .common_imports import *
from .diffusionSpectrum import diffusionSpectrum
from .utils import *


class TLIOModule(diffusionSpectrum):
    def __init__(self, config):
        super(TLIOModule, self).__init__(config)
        # self.loss = nn.MSELoss()
        if hasattr(self, "special_loss"):
            del self.special_loss
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "VAE"):
            del self.VAE
        if hasattr(self, "scheduler"):
            del self.scheduler

        # Calculate in_dim for TILO model
        # ResNet1D downsamples by 32
        window_size = self.config["window_size"]
        self.config["arch"] = "resnet"
        # Input dim 6 (acc+gyr), output dim 2 (x, y)
        self.model = get_model(
            self.config["arch"],
            net_config={"in_dim": window_size // 32 + 1},
            input_dim=6,
            output_dim=3,
        )
        self.toXYZ = nn.Sequential()

    def get_XY(self, x, y):
        # x_swap = torch.cat([x[:, 3:6, :], x[:, 0:3, :]], dim=1)
        return x, y.sum(dim=-1) / self.config["sampling_rate"]

    @overrides
    def regular_forward(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]

        batch_size, channels = x.shape[:2]
        x, y, _ = self.sample_noise((x, y), 0, dataL=L)
        x, y = self.get_XY(x=x, y=y)

        # TILO model returns mean, logstd
        mean, logstd = self.model(x)
        # print(cl.Fore.yellow + f"mean shape: {mean.shape}" + cl.Style.reset)
        # print(cl.Fore.yellow + f"logstd shape: {logstd.shape}" + cl.Style.reset)
        y_hat = mean

        assert torch.isfinite(y_hat).all(), "The estimate is not finite"
        assert torch.isfinite(logstd).all(), "The logstd is not finite"
        y_hat, y = self.postprocessing((y_hat, y[:, :3], L))

        losses = {
            f"loss_total/{mode}": get_loss(y_hat, logstd, y, self.current_epoch).mean(),
        }

        y_hat, y = y_hat.unsqueeze(-1), y.unsqueeze(-1)
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
        # x, y, L = self.get_XY(batch)
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]

        batch_size, channels = x.shape[:2]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L, do_noise=False)

        x, y = self.get_XY(x=x, y=y)

        mean, logstd = self.model(x)
        y_hat = mean

        y_hat, y = self.postprocessing((y_hat, y[:, :3], L))
        losses = {
            f"loss_total/{mode}": get_loss(y_hat, logstd, y, self.current_epoch).mean(),
        }

        y_hat, y = y_hat.unsqueeze(-1), y.unsqueeze(-1)
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

        # naive rescale and adjust for the callback function
        y_hat_dummy_zeros = torch.zeros(
            *(list(y_hat.shape[:2]) + [self.config["window_size"]]),
            device=y_hat.device,
            dtype=y_hat.dtype,
        )
        y_dummy_zeros = torch.zeros_like(
            y_hat_dummy_zeros, device=y.device, dtype=y.dtype
        )

        batch_indices = torch.arange(batch_size, device=y_hat.device)
        y_hat_dummy_zeros[batch_indices, :, L - 1] = (
            y_hat[batch_indices, :, 0] * self.config["sampling_rate"]
        )
        y_dummy_zeros[batch_indices, :, L - 1] = (
            y[batch_indices, :, 0] * self.config["sampling_rate"]
        )

        return losses[f"loss_total/{mode}"], {
            "dataX": y_hat_dummy_zeros,
            "dataY": y_dummy_zeros,
            "raw_dataX": x,  # Added raw input for filter
            "raw_mean": mean,  # Added raw mean for filter
            "raw_logstd": logstd,
            "raw_dataY": batch["dataY"],  # Added raw ground truth for filter evaluation
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
                sampling_rate=1,
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_X/{suffix}": NaiveDistanceError(
                channel_index=[0],
                sampling_rate=1,
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_Y/{suffix}": NaiveDistanceError(
                channel_index=[1],
                sampling_rate=1,
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError(
                channel_index=[2],
                sampling_rate=1,
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error/{suffix}": NaiveDistanceError(
                channel_index=[0, 1, 2],
                sampling_rate=1,
                avg_win=self.config["target_duration"],
            ),
        }

        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics

    @overrides
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.numel() > 0]

        optimizer = torch.optim.Adam(params, self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=3, eps=1e-12
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss_total/val",
                "interval": "epoch",
                "frequency": 1,
            },
        }
