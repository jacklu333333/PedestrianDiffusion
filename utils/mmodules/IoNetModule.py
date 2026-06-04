from .common_imports import *
from .diffusionSpectrum import diffusionSpectrum
from .utils import *


class IoNetModule(diffusionSpectrum):
    def __init__(self, config):
        super(IoNetModule, self).__init__(config)
        # Override the loss function and model from the base class
        if hasattr(self, "special_loss"):
            del self.special_loss
        self.loss_fn = CustomMultiLoss(nb_outputs=2)
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "VAE"):
            del self.VAE
        if hasattr(self, "scheduler"):
            del self.scheduler

        self.model = IoNet(window_size=config["window_size"])
        self.toXYZ = nn.Sequential()

    def get_XY(self, x, y):
        """
        Separates the input into two streams for IoNet and prepares the two ground truth outputs.
        """
        # IoNet takes two inputs: gyro (channels 3,4,5) and acc (channels 0,1,2)
        x1 = x[:, 3:6, :].swapaxes(1, 2)  # Gyroscope
        x2 = x[:, 0:3, :].swapaxes(1, 2)  # Accelerometer

        # IoNet predicts two outputs: translation vector (3D) and rotation quaternion (4D)
        # Assuming the ground truth 'y' contains this information directly.
        # This part might need adjustment based on how your ground truth is structured.
        # For this example, let's assume y is structured as [t_x, t_y, t_z, q_w, q_x, q_y, q_z]
        # and we need to split it.
        # If y is just [dx, dy, dtheta], this will need significant changes.
        # Based on the IoNet architecture, it expects a 3D vector and a 4D vector.
        # y1_true = (
        #     y[:, :3, -1] - y[:, :3, 0]
        # )  # Translation target (assuming it's a single vector per window)
        y1_true = y[:, 0:3].sum(dim=-1) / self.config["sampling_rate"]

        y2_true = y[:, 3:6]
        # convert the y2_true from euler to unit quaternion
        y2_true = angular_velocity_to_quaternion(
            y2_true, 1.0 / self.config["sampling_rate"]
        )

        return (x1, x2), (y1_true, y2_true)

    def convert_back_to_standard(self, y1_pred, y2_pred, L):
        """
        Converts IoNet outputs back to standard format if needed.
        """

        y1_pred = y1_pred.unsqueeze(-1)
        # Convert quaternion back to euler angles
        # y2_euler = quaternion_to_euler(y2_pred.unsqueeze(-1))
        y2_euler = quaternion_to_euler(y2_pred.unsqueeze(-1))
        # Combine translation and rotation back into a single tensor
        y_combined = torch.cat(
            [
                y1_pred,
                y2_euler,
            ],
            dim=-2,
        )

        # y_hat_pred_dummy_zeros = torch.zeros(
        #     *(list(y_combined.shape[:2]) + [self.config["window_size"]]),
        #     device=y_combined.device,
        # )
        # batch_size = y_combined.shape[0]
        # batch_indices = torch.arange(batch_size, device=y_combined.device)
        # y_hat_pred_dummy_zeros[batch_indices, :, L - 1] = y_combined[
        #     batch_indices, :, 0
        # ]

        return y_combined

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
        (x1, x2), (y1_true, y2_true) = self.get_XY(x=x, y=y)

        y1_pred, y2_pred = self.model(x1, x2)

        loss = self.loss_fn([y1_pred, y2_pred], [y1_true, y2_true])

        losses = {
            f"loss_total/{mode}": loss,
        }

        # Metrics would need to be adapted for position and quaternion outputs
        # For now, we will skip metric calculation as they are not directly compatible.
        y_hat = self.convert_back_to_standard(y1_pred, y2_pred, L)
        y = y.sum(dim=-1).unsqueeze(-1) / self.config["sampling_rate"]
        y_target = y
        metrics = {}
        for key, metric in self.metrics[mode].items():
            if (mode == "train" or mode == "val") and (
                "pearson" in key or "simVector" in key
            ):
                dictionary = metric(y_hat, y_target)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            elif "naive" in key:
                dictionary = metric(y_hat, y_target, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                dictionary = metric(y_hat, y_target)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v

        self.log_dict(
            dictionary=losses,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True if mode == "train" else False,
            sync_dist=True,
        )

        if mode == "train":
            self.log(
                name=f"lr/{mode}",
                value=self.trainer.optimizers[0].param_groups[0]["lr"],
                on_step=True,
                on_epoch=False,
                prog_bar=True,
                sync_dist=True,
            )
        self.log_dict(
            dictionary=metrics,
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True if mode == "train" else False,
            sync_dist=True,
        )

        return losses[f"loss_total/{mode}"]

    @overrides
    def sequential_forward(self, batch, mode="test"):
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        batch_size, channels = x.shape[:2]
        x, y, _ = self.sample_noise((x, y), 0, dataL=L, do_noise=False)
        (x1, x2), (y1_true, y2_true) = self.get_XY(x=x, y=y)

        y1_pred, y2_pred = self.model(x1, x2)

        loss = self.loss_fn([y1_pred, y2_pred], [y1_true, y2_true])

        losses = {
            f"loss_total/{mode}": loss,
        }

        y_hat = self.convert_back_to_standard(y1_pred, y2_pred, L)
        y = y.sum(dim=-1).unsqueeze(-1) / self.config["sampling_rate"]
        y_target = y
        metrics = {}
        for key, metric in self.metrics[mode].items():
            if (mode == "train" or mode == "val") and (
                "pearson" in key or "simVector" in key
            ):
                dictionary = metric(y_hat, y_target)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            elif "naive" in key:
                dictionary = metric(y_hat, y_target, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                dictionary = metric(y_hat, y_target)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v

        self.log_dict(
            dictionary=losses,
            on_step=False,
            on_epoch=True,
            prog_bar=True if mode == "test" else False,
            sync_dist=True if not mode == "peek_testing" else False,
        )
        self.log_dict(
            dictionary=metrics,
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

        batch_indices = torch.arange(batch_size, device=y_hat.device, dtype=torch.long)
        y_hat_dummy_zeros[batch_indices, :, L - 1] = (
            y_hat[batch_indices, :, 0] * self.config["sampling_rate"]
        )
        y_dummy_zeros[batch_indices, :, L - 1] = (
            y[batch_indices, :, 0] * self.config["sampling_rate"]
        )

        return losses[f"loss_total/{mode}"], {
            "dataX": y_hat_dummy_zeros,
            "dataY": y_dummy_zeros,
            "dataL": L,
            "name": batch["name"],
            "index": batch["index"],
            "label": batch["label"],
        }

    @overrides
    def _generate_metrics(self, suffix):
        # IoNet has its own complex loss, so standard metrics might not apply well.
        # Returning an empty dict to avoid errors.
        # metrics = {
        #     #
        #     f"metric_naive_distance_error_XY/{suffix}": NaiveDistanceError(
        #         channel_index=[0, 1],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_distance_error_X/{suffix}": NaiveDistanceError(
        #         channel_index=[0],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_distance_error_Y/{suffix}": NaiveDistanceError(
        #         channel_index=[1],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError(
        #         channel_index=[2],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_distance_error/{suffix}": NaiveDistanceError(
        #         channel_index=[0, 1, 2],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     # gyr
        #     f"metric_naive_Angular_error_X/{suffix}": NaiveAngularError(
        #         channel_index=[3],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_Angular_error_Y/{suffix}": NaiveAngularError(
        #         channel_index=[4],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_Angular_error_Z/{suffix}": NaiveAngularError(
        #         channel_index=[5],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        #     f"metric_naive_Angular_error/{suffix}": NaiveAngularError(
        #         channel_index=[3, 4, 5],
        #         sampling_rate=self.config["sampling_rate"],
        #         avg_win=self.config["target_duration"],
        #     ),
        # }
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
            # gyr
            f"metric_naive_Angular_error_X/{suffix}": NaiveAngularError(
                channel_index=[3],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_Angular_error_Y/{suffix}": NaiveAngularError(
                channel_index=[4],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_Angular_error_Z/{suffix}": NaiveAngularError(
                channel_index=[5],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_Angular_error/{suffix}": NaiveAngularError(
                channel_index=[3, 4, 5],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
        }

        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics

    @overrides
    def configure_optimizers(self):
        # Optimizer needs to see parameters from both the model and the custom loss
        params = list(self.model.parameters()) + list(self.loss_fn.parameters())
        optimizer = torch.optim.Adam(params, self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, eps=1e-12
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
