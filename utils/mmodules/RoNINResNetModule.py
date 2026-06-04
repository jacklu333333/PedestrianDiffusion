from utils.ronin.source.ronin_resnet import get_model

from .common_imports import *
from .diffusionSpectrum import diffusionSpectrum
from .utils import *


class RoNINResNetModule(diffusionSpectrum):
    def __init__(self, config):
        super(RoNINResNetModule, self).__init__(config)
        self.loss = nn.MSELoss()
        if hasattr(self, "special_loss"):
            del self.special_loss
        if hasattr(self, "model"):
            del self.model
        if hasattr(self, "VAE"):
            del self.VAE
        if hasattr(self, "scheduler"):
            del self.scheduler

        _fc_config = {"fc_dim": 512, "in_dim": 4, "dropout": 0.5, "trans_planes": 128}
        self.model = get_model("resnet101", _output_channel=3, _fc_config=_fc_config)
        self.toXYZ = nn.Sequential()

    def get_XY(self, x, y):
        return x, y[:, :3].sum(dim=-1) / self.config["sampling_rate"]
        # return x, y[:, :2, -1] - y[:, :2, 0]

    def forward(self, x):
        return self.model(x)

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
        # print(cl.Fore.red, f"y: {y.shape} x: {x.shape}", cl.Style.reset)
        y_hat = self.forward(x)
        # print(
        #     cl.Fore.red,
        #     f"y_hat: {y_hat.shape}, y: {y.shape} x: {x.shape}",
        #     cl.Style.reset,
        # )
        assert torch.isfinite(y_hat).all(), "The estimate is not finite"
        # y_hat, y = self.postprocessing((y_hat, y, L))

        losses = {
            f"loss_total/{mode}": self.loss(y_hat, y),
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
        """
        TODO: Implement the PipeLine approach for speed up
        """
        # pipeline = diffusers.DiffusionPipeline(
        # )
        x, y = self.get_XY(x=x, y=y)

        y_hat = self.forward(x)

        # y_hat, y = self.postprocessing((y_hat, y, L))
        losses = {
            f"loss_total/{mode}": self.loss(y_hat, y),
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
        # epand the last dim to window_size
        # y_hat = y_hat.expand(-1, -1, self.config["window_size"]) / self.config["window_size"]
        # y = y.expand(-1, -1, self.config["window_size"]) /self.config["window_size"]

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
            # "dataX": y_hat,
            # "dataY": y,
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
            # gyr
            # f"metric_mse_gyr_X/{suffix}": mMSE(channel_index=[3], norm=False),
            # f"metric_mse_gyr_Y/{suffix}": mMSE(channel_index=[4], norm=False),
            # f"metric_mse_gyr_Z/{suffix}": mMSE(channel_index=[5], norm=False),
            # f"metric_mse_gyr/{suffix}": mMSE(channel_index=[3, 4, 5], norm=False),
            # gyr
        }

        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics

    # # @overrides
    # def on_test_step(self, batch, batch_idx):
    #     loss, _ = self.sequential_forward(batch, mode="test")
    #     return loss

    # @overrides
    # def on_test_end(self):
    #     # Convert lists to tensors for gathering
    #     if len(self.estimation_result["dataX"]) > 0:
    #         for key, value in self.estimation_result.items():
    #             self.estimation_result[key] = torch.stack(value, dim=0)

    #         # Gather data from all processes
    #         if self.trainer.world_size > 1:
    #             gathered_result = {
    #                 key: self.all_gather(value).flatten(0, 1)
    #                 for key, value in self.estimation_result.items()
    #             }
    #         else:
    #             gathered_result = self.estimation_result

    #         # Only save on rank 0 to avoid duplicate saves
    #         if self.global_rank == 0:
    #             # Save or process the gathered data
    #             results = {key: value.cpu() for key, value in gathered_result.items()}
    #             log_dir = self.logger.log_dir
    #             unique_labels = (
    #                 torch.unique(results["label"])
    #                 if "label" in results
    #                 else torch.tensor([0])
    #             )
    #             ATE_error_total = {"pos": defaultdict(list), "ori": defaultdict(list)}
    #             RTE_error_total = {"pos": defaultdict(list), "ori": defaultdict(list)}

    #             for label in unique_labels:
    #                 # Get indices for this label
    #                 label_mask = (
    #                     results["label"] == label
    #                     if "label" in results
    #                     else torch.ones_like(results["name"], dtype=torch.bool)
    #                 )

    #                 # Get unique names for this label
    #                 label_names = results["name"][label_mask]
    #                 unique_names = torch.unique(label_names)

    #                 print(f"Processing label: {label.item()}")

    #                 for name in unique_names:
    #                     # Get indices for this specific (label, name) combination
    #                     name_mask = label_mask & (results["name"] == name)

    #                     # Extract data for this group
    #                     est = results["dataX"][name_mask]
    #                     targ = results["dataY"][name_mask]
    #                     length = results["dataL"][name_mask]
    #                     indexes = results["index"][name_mask]
    #                     ns = results["name"][name_mask]

    #                     # Process filename

    #                     str_name = (
    #                         self.trainer.datamodule.test_dataset.datasets[label].files[
    #                             name
    #                         ]
    #                         if len(unique_labels) > 1
    #                         else self.trainer.datamodule.test_dataset.files[name]
    #                     )

    #                     if "OIOD" in INV_DATASET_DICT[label.item()]:
    #                         str_name = str_name[1]
    #                     print(str_name)
    #                     str_name = (
    #                         str(str_name)
    #                         .replace("/", "_")
    #                         .replace(" ", "_")
    #                         .replace("[", "")
    #                         .replace("]", "")
    #                     )

    #                     # Create hierarchical save path: label/name
    #                     label_n = INV_DATASET_DICT[label.item()]
    #                     save_path = Path(log_dir) / f"label_{label_n}" / f"{str_name}"
    #                     save_path.mkdir(parents=True, exist_ok=True)

    #                     # Sort by indexes
    #                     sorted_indices = torch.argsort(indexes)
    #                     est = est[sorted_indices]
    #                     targ = targ[sorted_indices]
    #                     length = length[sorted_indices]
    #                     indexes = indexes[sorted_indices]
    #                     ns = ns[sorted_indices]

    #                     # Save tensors
    #                     torch.save(est, save_path / "estimates.pt")
    #                     torch.save(targ, save_path / "targets.pt")
    #                     torch.save(length, save_path / "lengths.pt")
    #                     torch.save(indexes, save_path / "indexes.pt")
    #                     torch.save(ns, save_path / "names.pt")

    #                     # Process sequences by length
    #                     temp_est = est.swapaxes(0, 1).cumsum(dim=1) * (
    #                         1 / self.config["sampling_rate"]
    #                     )
    #                     temp_targ = targ.swapaxes(0, 1).cumsum(dim=1) * (
    #                         1 / self.config["sampling_rate"]
    #                     )
    #                     torch.save(temp_est, save_path / "integral_estimates.pt")
    #                     torch.save(temp_targ, save_path / "integral_targets.pt")

    #                     for idxm, metri in enumerate(["pos"]):
    #                         rte = np.mean(np.linalg.norm(temp_est - temp_targ, axis=0))
    #                         ate = np.mean(
    #                             np.linalg.norm(
    #                                 temp_est[:, -1] - temp_targ[:, -1], axis=0
    #                             )
    #                         )
    #                         ATE_error_total[metri][label_n].append(ate)
    #                         RTE_error_total[metri][label_n].append(rte)

    #             for key in ATE_error_total.keys():
    #                 for k in ATE_error_total[key].keys():
    #                     self.logger.log_metrics(
    #                         {
    #                             f"ATE_{key}_{k}": np.mean(ATE_error_total[key][k]),
    #                             f"RTE_{key}_{k}": np.mean(RTE_error_total[key][k]),
    #                         },
    #                         step=self.current_epoch,
    #                     )
    #                     print(
    #                         cl.Fore.red,
    #                         f"ATE {key} {k} error: {np.mean(ATE_error_total[key][k])}",
    #                         f"RTE {key} {k} error: {np.mean(RTE_error_total[key][k])}",
    #                         cl.Style.reset,
    #                     )

    #     return super(pl.LightningModule, self).on_test_end()

    @overrides
    def configure_optimizers(self):
        params = [p for p in self.model.parameters() if p.numel() > 0]
        num_devices = self.trainer.world_size if self.trainer else 1
        # get the accumulate_grad_batches from the trainer
        accumulation = self.trainer.accumulate_grad_batches if self.trainer else 1
        num_batches = len(self.trainer.datamodule.train_dataloader())
        # valid_batches = num_batches // num_devices // accumulation
        valid_batches = num_batches // self.trainer.world_size
        # valid_batches = 1000

        # if self.config["ema"]["enable"]:
        #     self.ema = diffusers.training_utils.EMAModel(
        #         parameters=params,
        #         # decay=self.config["ema"]["ema_decay"],
        #         decay=1 - 1 / (valid_batches),
        #         # min_decay: float = 0.0,
        #         update_after_step=valid_batches,
        #         use_ema_warmup=True,
        #         inv_gamma=valid_batches,
        #         # power: Union[float, int] = 2 / 3,
        #         foreach=True,
        #         # model_cls: Optional[Any] = None,
        #         # model_config: Dict[str, Any] = None,
        #     )
        #     # self.ema = diffusers.training_utils.EMAModel(
        #     #     parameters=params,
        #     #     decay=self.config["ema"]["ema_decay"],
        #     # )
        #     self.ema.to(self.device)

        # optimizer = optim.AdamW(params, lr=self.lr, fused=True)
        # # build-in
        # # optimizer = optim.Adam(params, lr=self.lr)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.9,
        #     patience=100,
        #     cooldown=50,
        #     verbose=False,
        #     min_lr=1e-8,
        # )

        optimizer = torch.optim.Adam(params, self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, factor=0.1, patience=10, eps=1e-12
        )

        # deepspeed version # DO NOT SUPPORT LR_FINDER
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(
        #     params,
        #     lr=self.lr,
        # )
        # scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
        #     optimizer=optimizer,
        #     num_warmup_steps=valid_batches * self.config["warm_up"],
        #     num_training_steps=valid_batches,
        #     # num_warmup_steps=self.warm_up,
        #     # num_training_steps=num_batches,
        #     # num_cycles=0.3,
        # )
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": scheduler,
        #         "monitor": "loss_total/train_step",
        #         "interval": "step",
        #         "frequency": self.config["accumulate_grad_batches"],
        #     },
        # }

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss_total/val",
                "interval": "epoch",
                "frequency": 1,
            },
        }
