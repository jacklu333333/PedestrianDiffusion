from diffusers.models import (
    UNet2DConditionModel,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
)

from .baseDiffusionModule import baseDiffusionModule
from .common_imports import *
from .utils import *


class PedestrianDiffusion(baseDiffusionModule):
    def __init__(self, config):
        _scaling = 5
        sensor_norm_config = {
            "acc_mean": 0,
            "acc_std": 5.78218412399292 * _scaling,
            "gyr_mean": 0,
            "gyr_std": 2.383342266082764 * _scaling,
        }
        label_norm_config = {
            "mean": 0,
            "std": 13.15231227874756 * _scaling,
            "ori_mean": 0,
            "ori_std": 2.802832841873169 * _scaling,
        }

        super(PedestrianDiffusion, self).__init__(config)

        self.freq_sim = nn.CosineSimilarity(dim=1)
        self.special_loss = spectrumMultiTaskLoss(dt=1.0 / self.config["sampling_rate"])
        if hasattr(self, "toObservation"):
            del self.toObservation
        self.toObservation = nn.Sequential(
            Rearrange("b c (f1 f2) t I -> b (c I) t f1 f2", I=2, f1=4, f2=4),
        )
        if hasattr(self, "deObservation"):
            del self.deObservation
        self.deObservation = nn.Sequential(
            Rearrange("b (c I) t f1 f2 -> b c (f1 f2) t I", I=2, f1=4, f2=4),
        )

        if hasattr(self, "sensorProcessor"):
            del self.sensorProcessor
        self.sensorProcessor = nn.Sequential(
            batchNormalizeSensor(**sensor_norm_config),
        )
        if hasattr(self, "finalSensorProcessor"):
            del self.finalSensorProcessor

        self.labelProcessor = nn.Sequential(
            bathNormalizeRelativePosNOri(**label_norm_config),
        )
        if hasattr(self, "finalLabelProcessor"):
            del self.finalLabelProcessor
        self.finalLabelProcessor = nn.Sequential(
            batchDenormalizeRelativePosNOri(**label_norm_config),
        )

        if hasattr(self, "VAE"):
            del self.VAE
        # self.VAE = mVAEModel(self.config["latent_dim"], eps=1e-14)
        self.VAE = VAE3D_1024(
            latent_dim=self.config["latent_dim"],
            in_channels=6,
        )
        self.VAE.deleteDecoder()
        for param in self.VAE.parameters():
            param.requires_grad = False

        self.model = UNet3DConditionModel(
            sample_size=(4, 4),
            in_channels=24,
            out_channels=12,
            down_block_types=(
                "CrossAttnDownBlock3D",
                "CrossAttnDownBlock3D",
                # "CrossAttnDownBlock3D",
                # "CrossAttnDownBlock3D",
            ),
            up_block_types=(
                "CrossAttnUpBlock3D",
                "CrossAttnUpBlock3D",
                # "CrossAttnUpBlock3D",
                # "CrossAttnUpBlock3D",
            ),
            block_out_channels=(
                np.array(
                    [128, 256],
                    dtype=int,
                )
            ).tolist(),
            layers_per_block=1,
            # downsample_padding: int = 1,
            # mid_block_scale_factor: float = 1,
            # act_fn: str = "silu",
            # norm_num_groups: Optional[int] = 32,
            # norm_eps=1e-7,
            cross_attention_dim=self.config["latent_dim"],
            attention_head_dim=64,
            # num_attention_heads=(4, 8, 8, 8),
            # time_cond_proj_dim: Optional[int] = None,
        )
        del self.scheduler
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config["num_time_steps"],
            beta_schedule=self.config["scheduler_mode"],
            beta_start=self.config["scheduler_s"],
            beta_end=self.config["scheduler_e"],
            solver_order=2,
            prediction_type="sample",
            lower_order_final=True,
            rescale_betas_zero_snr=True,
        )
        # self.scheduler = DPMSolverSinglestepScheduler(
        #     num_train_timesteps=self.config["num_time_steps"],
        #     beta_schedule=self.config["scheduler_mode"],
        #     beta_start=self.config["scheduler_s"],
        #     beta_end=self.config["scheduler_e"],
        #     solver_order=1,
        #     # thresholding=True,
        #     # trained_betas=gen_linear_beta_t(
        #     #     num_timesteps=self.config["num_time_steps"]
        #     # ),
        #     sample_max_value=torch.finfo(torch.float32).max,
        #     prediction_type="v_prediction",
        #     variance_type="v_prediction",
        #     final_sigmas_type="zero",
        #     lower_order_final=True,
        #     # rescale_betas_zero_snr=True,
        # )

        # self.scheduler = EulerDiscreteScheduler(
        #     num_train_timesteps=self.config["num_time_steps"],
        #     beta_start=self.config["scheduler_s"],
        #     beta_end=self.config["scheduler_e"],
        #     beta_schedule=self.config["scheduler_mode"],
        #     prediction_type="v_prediction",  # or "epsilon", match your training
        #     rescale_betas_zero_snr=True,
        # )

        # self.scheduler = DDIMScheduler(
        #     num_train_timesteps=self.config["num_time_steps"],
        #     beta_start=self.config["scheduler_s"],
        #     beta_end=self.config["scheduler_e"],
        #     beta_schedule=self.config["scheduler_mode"],
        #     prediction_type="v_prediction",  # or "epsilon", match your training
        #     # rescale_betas_zero_snr=True,
        # )

    def generate_timesteps(self, batch_size: int, target_time_steps: int, device=None):
        """
        Generate a tensor of timesteps such that:
        - Every integer from 0 to target_time_steps - 1 appears at least once
        - The tensor length equals batch_size
        - The order is randomized

        Args:
            batch_size (int): Number of samples to generate
            target_time_steps (int): Upper bound (exclusive) of timestep values
            device (torch.device, optional): Device to place the tensor on

        Returns:
            torch.Tensor: Shuffled tensor of timesteps
        """
        if batch_size < target_time_steps:
            raise ValueError("batch_size must be >= target_time_steps")

        # Ensure coverage of all timesteps
        base = torch.arange(target_time_steps, device=device)

        # Fill remaining slots with random values
        extra = torch.randint(
            0, target_time_steps, (batch_size - target_time_steps,), device=device
        )

        # Concatenate and shuffle
        timesteps = torch.cat([base, extra])
        timesteps = timesteps[torch.randperm(batch_size, device=device)]

        return timesteps

    def on_validation_start(self):
        if hasattr(self, "ema"):
            # print("Loading EMA weights for validation...")
            self.ema.store(self.parameters())  # 1. Backup raw weights
            self.ema.copy_to(self.parameters())  # 2. Load smooth EMA weights

    def on_test_start(self):
        super().on_test_start()  # Keep your existing print logic
        ckpt_path = getattr(self.trainer, "ckpt_path", None)
        if ckpt_path:
            rank_zero_info(
                cl.Fore.green
                + f"Testing with weights explicitly loaded from: {ckpt_path}"
                + cl.Style.reset
            )
        else:
            rank_zero_info(
                cl.Fore.yellow
                + "Testing with in-memory weights (explicit ckpt_path not provided)."
                + cl.Style.reset
            )
            if hasattr(self, "ema"):
                rank_zero_info(
                    cl.Fore.cyan + "Loading EMA weights for testing..." + cl.Style.reset
                )
                self.ema.store(self.parameters())  # 1. Backup raw weights
                self.ema.copy_to(self.parameters())  # 2. Load smooth EMA weights
            else:
                # raise ValueError(
                #     "EMA is not enabled, and no checkpoint path provided for testing."
                # )
                rank_zero_info(
                    cl.Fore.YELLOW
                    + "EMA is not enabled, and no checkpoint path provided for testing."
                    + cl.Style.reset
                )

        rank_zero_info(
            cl.Fore.green
            + f"Testing with inferencing step of {self.config['target_time_steps']} steps"
            + cl.Style.reset
        )

    def on_validation_end(self):
        if hasattr(self, "ema"):
            # print("Restoring raw weights after validation...")
            self.ema.restore(
                self.parameters()
            )  # 3. Restore raw weights for next training step

    def on_save_checkpoint(self, checkpoint):
        super().on_save_checkpoint(checkpoint)
        if hasattr(self, "ema"):
            self.ema.store(self.parameters())
            self.ema.copy_to(self.parameters())
            checkpoint["state_dict"] = {
                k: v.clone() for k, v in self.state_dict().items()
            }
            self.ema.restore(self.parameters())

    def on_train_epoch_end(self):
        if hasattr(self, "ema") and self.config["ema"]["enable"]:
            self.ema.store(self.parameters())  # 1. Backup raw weights
            self.ema.copy_to(self.parameters())  # 2. Load smooth EMA weights
            super().on_train_epoch_end()
            self.ema.restore(self.parameters())  # 3. Restore raw weights
        else:
            super().on_train_epoch_end()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if hasattr(self, "ema") and self.config["ema"]["enable"]:
            self.ema.step(self.parameters())
        return super().on_train_batch_end(outputs, batch, batch_idx)

    @rank_zero_only
    def peeking(self):
        if not hasattr(self.trainer.datamodule, "test_dataset"):
            self.trainer.datamodule.setup("test")
        dataloader = self.trainer.datamodule.test_dataloader()
        self.eval()
        outputs = []
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Peek Test")):
                batch = self.transfer_batch_to_device(batch, self.device, batch_idx)
                out = self.sequential_forward(batch, mode="peek_testing")
                outputs.append(out)

        self.on_test_epoch_end()

    def on_test_start(self):
        self.estimation_result = {
            "dataX": [],
            "dataY": [],
            "dataL": [],
            "name": [],
            "index": [],
            "label": [],
        }
        return super().on_test_start()

    def training_step(self, batch, batch_idx):
        return self.regular_forward(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.regular_forward(batch, "val")

    def test_step(self, batch, batch_idx):
        loss, result_dict = self.sequential_forward(batch, mode="test")
        return {"loss": loss, "results": result_dict}

    def predict_step(self, batch, batch_idx):
        return self.sequential_forward(batch, mode="predict")[1]

    def forward_model(
        self,
        x: torch.Tensor,
        original: torch.tensor,
        t: torch.Tensor,
        encoding: torch.tensor,
        attention_mask: torch.tensor,
        class_labels,
    ):
        out = self.toObservation(x)

        out = self.model(
            sample=out,
            # hidden_states=x_,
            timestep=t,
            # class_labels=class_labels,
            encoder_hidden_states=encoding,
            attention_mask=attention_mask,
            # global_hidden_states=encoding,
            # added_time_ids=torch.arange(0, out.shape[2], device=out.device)
            # .long()
            # .unsqueeze(0)
            # .expand(out.shape[0], -1),
        ).sample

        assert torch.isfinite(
            out
        ).all(), 'The estimate is not finite in the function "forward_model"'

        out = self.deObservation(out)
        return out

    def compute_loss(
        self,
        freq_estimate,
        freq_target,
        time_estimate,
        time_target,
        timesteps,
        mode,
    ):
        # batch_size, channels = freq_target.shape[:2]
        # freq_loss = torch.vmap(self.freq_loss)(
        #     freq_estimate * 1e0, freq_target * 1e0
        # ).mean()
        # time_loss = torch.vmap(self.time_loss)(
        #     time_estimate / self.config["sampling_rate"] * 1e0,
        #     time_target / self.config["sampling_rate"] * 1e0,
        # ).mean()
        # time_sum_loss = torch.vmap(self.time_loss)(
        #     time_estimate.cumsum(dim=-1) / self.config["sampling_rate"] * 1e0,
        #     time_target.cumsum(dim=-1) / self.config["sampling_rate"] * 1e0,
        # )

        # compute where step = 0
        # aux_mask = (timesteps - 1).clamp(min=0) == 0
        # prim = time_sum_loss.clone()[aux_mask]
        # time_sum_loss = time_sum_loss.mean()

        # loss_aux = self.time_sum_loss(
        #     prim, torch.zeros_like(prim)
        # ) / aux_mask.sum().clamp(min=1)
        # # total_loss = freq_loss + time_loss + time_sum_loss
        # # total_loss = time_loss
        # # total_loss = freq_loss

        # freq_sim = torch.vmap(self.freq_sim.forward)(
        #     freq_estimate.reshape(batch_size, channels, -1),
        #     freq_target.reshape(batch_size, channels, -1),
        # )
        # freq_sim = freq_sim.mean()
        # sign = torch.sign(freq_sim)
        # freq_sim = (
        #     torch.pow(
        #         freq_sim.abs().clamp(min=0, max=1),
        #         timesteps.reshape(freq_sim.shape[0], 1, 1) + 1,
        #     )
        #     .mul(sign)
        #     .mean()
        # )
        # assert torch.isfinite(freq_sim).all(), "The frequency similarity is not finite"

        # time_sim = torch.vmap(self.freq_sim.forward)(
        #     time_estimate.reshape(time_estimate.shape[0], channels, -1),
        #     time_target.reshape(time_target.shape[0], channels, -1),
        # )
        # time_sim = time_sim.mean()
        # sign = torch.sign(time_sim)
        # time_sim = (
        #     torch.pow(
        #         time_sim.abs().clamp(min=0, max=1),
        #         timesteps.reshape(time_sim.shape[0], 1, 1) + 1,
        #     )
        #     .mul(sign)
        #     .mean()
        # )
        # assert torch.isfinite(time_sim).all(), "The time similarity is not finite"

        losses = self.special_loss(
            y_hat_f=freq_estimate,
            y_f=freq_target,
            y_hat_t=time_estimate,
            y_t=time_target,
            timesteps=timesteps,
        )

        # return {
        #     f"loss_freq/{mode}": freq_loss,
        #     f"loss_time/{mode}": time_loss,
        #     f"loss_time_sum/{mode}": time_sum_loss,
        #     f"loss_total/{mode}": total_loss,
        #     f"loss_aux/{mode}": loss_aux,
        #     f"freq_sim/{mode}": freq_sim,
        #     f"time_sim/{mode}": time_sim,
        #     f"freq_sim_c/{mode}": 1 - freq_sim,
        #     f"time_sim_c/{mode}": 1 - time_sim,
        # }
        for key in list(losses.keys()):
            losses[f"{key}/{mode}"] = losses.pop(key)
        # losses[f"freq_sim/{mode}"] = freq_sim
        # losses[f"time_sim/{mode}"] = time_sim
        # losses[f"freq_sim_c/{mode}"] = 1 - freq_sim
        # losses[f"time_sim_c/{mode}"] = 1 - time_sim
        return losses

    def _generate_noise(self, x, y):
        return torch.randn_like(y)
        # return x

    def _generate_target(self, y, noise, timesteps):
        target = self.add_noise(
            original=y,
            noise=noise,
            t=(timesteps - 1).clamp(0, self.config["target_time_steps"]),
        )
        return target

    def regular_forward(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        # return self.TVLB(batch, mode)
        return self.VLB(batch, mode)

    def TVLB(
        self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor], mode: str
    ) -> torch.Tensor:
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        encoding = batch["encoding"]

        batch_size, channels = x.shape[:2]
        x, y, _ = self.sample_noise((x, y), 0, dataL=L)

        latent_space = self.VAE.encode(rearrange(x, "b c f t I -> b c (I f) t"))
        latent_space = rearrange(latent_space, "b c w h -> b c (w h)")
        encoding = torch.cat([latent_space, encoding], dim=1)
        # encoding = latent_space.view(
        #     x.shape[0], self.config["latent_dim"], -1
        # ).swapaxes(-1, -2)

        # # Classifier-Free Guidance: randomly drop conditioning
        # if mode == "train" and self.config.get("cfg_drop_prob", 0.0) > 0:
        #     unconditional_encoding = torch.zeros_like(encoding)
        #     drop_mask = (
        #         torch.rand(batch_size, device=self.device)
        #         < self.config["cfg_drop_prob"]
        #     )
        #     encoding[drop_mask] = unconditional_encoding[drop_mask]

        timesteps = torch.randint(
            0,
            self.config["target_time_steps"],
            (x.shape[0],),
            requires_grad=False,
            device=x.device,
        )
        # # guarantee at least one sample with t=0 in each batch
        # if mode == "train":
        #     # timesteps[torch.randint(0, timesteps.shape[0], (1,))] = 1
        #     a, b = np.random.choice(timesteps.shape[0], 2, replace=False)
        #     timesteps[a] = 1
        #     timesteps[b] = self.config["target_time_steps"] - 1

        noise = self._generate_noise(x, y)

        pseudo_observation = self.add_noise(
            original=y,
            noise=noise,
            t=timesteps,
        )

        estimate = self.forward_model(
            x=torch.cat([pseudo_observation, x], dim=1),
            original=None,
            t=timesteps,
            encoding=encoding,
            class_labels=batch["datasets"],
        )
        assert (
            estimate.shape == pseudo_observation.shape
        ), f"Shape mismatch: estimate {estimate.shape}, pseudo_observation {pseudo_observation.shape}"
        assert torch.isfinite(estimate).all(), "The estimate is not finite"

        y_hat = self.step_backward(
            original_input=pseudo_observation,
            estimate_noise=estimate,
            t=timesteps,
            target_steps=self.config["target_time_steps"],
            get_x0=False,
            different_timestamp=True,
        )
        target = self._generate_target(y, noise, timesteps)

        y_hat_t, target_t = self.postprocessing((y_hat, target, L))

        # aux_mask = timesteps == 0
        # y_hat_t_mask = y_hat_t[aux_mask]
        # target_t_mask = target_t[aux_mask]

        # if y_hat_t_mask.shape[0] > 0:
        #     y_hat_t_mask = self.finalSMoother(y_hat_t_mask)

        losses = self.compute_loss(
            freq_estimate=y_hat,
            freq_target=target,
            time_estimate=y_hat_t,
            time_target=target_t,
            timesteps=timesteps,
            mode=mode,
        )

        masks = ~self.mask_generation(
            dataL=L,
            window_size=self.config["window_size"],
            batch_size=batch_size,
            channels=channels,
        )
        # masks = self.video_mask_generation(
        #     dataL=L,
        #     window_size=self.config["window_size"],
        #     batch_size=batch_size,
        #     channels=channels,
        # )

        y_hat_t = y_hat_t.detach() * masks
        target_t = target_t.detach() * masks
        metrics = {}
        for key, metric in self.metrics[mode].items():
            if (mode == "train" or mode == "val") and (
                "pearson" in key or "simVector" in key
            ):
                dictionary = metric(y_hat_t, target_t, steps=timesteps)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            elif "naive" in key:
                # metrics[key] = metric(y_hat_t, target_t, dataL=L)
                dictionary = metric(y_hat_t, target_t, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                # metrics[key] = metric(y_hat_t, target_t)
                dictionary = metric(y_hat_t, target_t)
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

    def VLB(
        self,
        batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        mode: str,
    ) -> torch.Tensor:
        # raise NotImplementedError("VLB is not implemented yet.")
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        encoding = batch["encoding"]
        attention_mask = batch["attention_mask"]

        batch_size, channels = x.shape[:2]
        x, y, _ = self.sample_noise((x, y), 0, dataL=L)

        latent_space = self.VAE.encode(x).swapaxes(-1, -2)
        encoding = torch.cat([latent_space, encoding], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(attention_mask.shape[0], 1, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )

        # latent_space = self.VAE.encode(self.toObservation(x))
        # encoding = latent_space.view(
        #     x.shape[0], self.config["latent_dim"], -1
        # ).swapaxes(-1, -2)

        # Classifier-Free Guidance: randomly drop conditioning
        # if mode == "train" and self.config.get("cfg_drop_prob", 0.0) > 0:
        #     unconditional_encoding = torch.zeros_like(encoding)
        #     drop_mask = (
        #         torch.rand(batch_size, device=self.device)
        #         < self.config["cfg_drop_prob"]
        #     )
        #     encoding[drop_mask] = unconditional_encoding[drop_mask]

        timesteps = torch.randint(
            0,
            self.config["target_time_steps"],
            (x.shape[0],),
            requires_grad=False,
            device=x.device,
        )
        # timesteps = self.generate_timesteps(
        #     batch_size=x.shape[0],
        #     target_time_steps=self.config["target_time_steps"],
        #     device=self.device,
        # )
        # # guarantee at least one sample with t=0 in each batch
        # if mode == "train":
        #     # timesteps[torch.randint(0, timesteps.shape[0], (1,))] = 1
        #     a, b = np.random.choice(timesteps.shape[0], 2, replace=False)
        #     timesteps[a] = 1
        #     timesteps[b] = self.config["target_time_steps"] - 1

        noise = self._generate_noise(x, y)

        pseudo_observation = self.add_noise(
            original=y,
            noise=noise,
            t=timesteps,
        )

        estimate = self.forward_model(
            x=torch.cat([pseudo_observation, x], dim=1),
            # x=pseudo_observation,
            original=None,
            t=timesteps,
            encoding=encoding,
            attention_mask=attention_mask,
            class_labels=batch["datasets"],
        )
        assert (
            estimate.shape == pseudo_observation.shape
        ), f"Shape mismatch: estimate {estimate.shape}, pseudo_observation {pseudo_observation.shape}"
        assert torch.isfinite(estimate).all(), "The estimate is not finite"

        # y_hat = self.step_backward(
        #     original_input=pseudo_observation,
        #     estimate_noise=estimate,
        #     t=timesteps,
        #     target_steps=self.config["target_time_steps"],
        #     get_x0=False,
        #     different_timestamp=True,
        # )
        # # target = self._generate_target(y, noise, timesteps)
        target = y

        # y_hat = scheduler_to_X0(
        #     scheduler=self.scheduler,
        #     noisy=pseudo_observation,
        #     noise=estimate,
        #     t=timesteps,
        # )
        y_hat = estimate
        y_hat_t, target_t = self.postprocessing((y_hat, target, L))

        losses = self.compute_loss(
            freq_estimate=y_hat,
            freq_target=target,
            #
            time_estimate=y_hat_t,
            time_target=target_t,
            #
            timesteps=timesteps,
            mode=mode,
        )

        # aux_mask = timesteps == 0
        # y_hat_t_mask = y_hat_t[aux_mask]
        # target_t_mask = target_t[aux_mask]

        # if y_hat_t_mask.shape[0] > 0:
        #     y_hat_t_mask = self.finalSMoother(y_hat_t_mask)

        masks = ~self.mask_generation(
            dataL=L,
            window_size=self.config["window_size"],
            batch_size=batch_size,
            channels=channels,
        )
        # masks = self.video_mask_generation(
        #     dataL=L,
        #     window_size=self.config["window_size"],
        #     batch_size=batch_size,
        #     channels=channels,
        # )

        y_hat_t = y_hat_t.detach() * masks
        target_t = target_t.detach() * masks
        metrics = {}
        for key, metric in self.metrics[mode].items():
            if (mode == "train" or mode == "val") and (
                "pearson" in key or "simVector" in key
            ):
                dictionary = metric(y_hat_t, target_t)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            elif "naive" in key:
                # metrics[key] = metric(y_hat_t, target_t, dataL=L)
                dictionary = metric(y_hat_t, target_t, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                # metrics[key] = metric(y_hat_t, target_t)
                dictionary = metric(y_hat_t, target_t)
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

    def sequential_forward(self, batch, mode="test"):
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        encoding = batch["encoding"]
        attention_mask = batch["attention_mask"]
        batch_size, channels = x.shape[:2]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L, do_noise=False)
        batch_size, channels = x.shape[:2]
        """
        TODO: Implement the PipeLine approach for speed up
        """
        # pipeline = diffusers.DiffusionPipeline(
        # )
        latent_space = self.VAE.encode(x).swapaxes(-1, -2)
        encoding = torch.cat([latent_space, encoding], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(attention_mask.shape[0], 1, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )

        target_testing_steps = (
            self.config["num_time_steps"]
            if mode != "test"
            else self.config["target_time_steps"]
        )
        self.scheduler.set_timesteps(target_testing_steps)
        # timesteps = torch.arange(
        #     start=target_testing_steps - 1,
        #     end=-1,
        #     step=-1,
        #     device=x.device,
        # )
        # latent_space = self.VAE.encode(self.toObservation(x))
        # encoding = latent_space.view(
        #     x.shape[0], self.config["latent_dim"], -1
        # ).swapaxes(-1, -2)
        unconditional_encoding = torch.zeros_like(encoding)
        cfg_scale = self.config.get("cfg_scale", 1.0)

        estimate_result = self._generate_noise(x, y)
        for idx, t in tqdm(
            enumerate(self.scheduler.timesteps),
            total=len(self.scheduler.timesteps),
            desc="Sampling",
            leave=False,
        ):
            assert torch.isfinite(
                estimate_result
            ).all(), f"The estimate is not finite at step: {t}"
            # Duplicate input for conditional and unconditional passes
            # estimate_result = torch.cat([estimate_result, x], dim=1)
            # model_input = torch.cat([estimate_result] * 2)
            # model_timestep = torch.cat([step_t] * 2)
            # model_encoding = torch.cat([unconditional_encoding, conditional_encoding])
            estimate_result = self.scheduler.scale_model_input(estimate_result, t)

            estimate = self.forward_model(
                # x=model_input,
                # original=None,
                # t=model_timestep,
                # encoding=model_encoding,
                x=torch.cat([estimate_result, x], dim=1),
                # x=estimate_result,
                original=None,
                t=t.unsqueeze(0).repeat(batch_size).to(self.device),
                class_labels=batch["datasets"],
                encoding=encoding,
                attention_mask=attention_mask,
            )

            # Split predictions and apply guidance
            # unconditional_pred, conditional_pred = estimate.chunk(2)
            # estimate = unconditional_pred + cfg_scale * (
            #     conditional_pred - unconditional_pred
            # )

            # assert torch.isfinite(estimate).all()
            estimate_result = self.step_backward(
                original_input=estimate_result,
                estimate_noise=estimate,
                t=t.unsqueeze(0).repeat(batch_size),
                target_steps=target_testing_steps,
                # get_x0=False if t > 0 else True,
                # get_x0=False,
            )

        estimate_result_t, y_t = self.postprocessing((estimate_result, y, L))
        # estimate_result_t = self.finalSMoother(estimate_result_t)
        losses = self.compute_loss(
            freq_estimate=estimate_result,
            freq_target=y,
            time_estimate=estimate_result_t,
            time_target=y_t,
            timesteps=torch.zeros(
                estimate_result.shape[0], device=estimate_result.device
            ),
            mode=mode,
        )

        masks = ~self.mask_generation(
            dataL=L,
            window_size=self.config["window_size"],
            batch_size=batch_size,
            channels=channels,
        )

        estimate_result_t = estimate_result_t.detach() * masks
        y_t = y_t.detach() * masks

        metrics = {}
        for key, metric in self.metrics[mode].items():
            if "naive" in key:
                # metrics[key] = metric(estimate_result_t, y_t, dataL=L)
                dictionary = metric(estimate_result_t, y_t, dataL=L)
                for k, v in dictionary.items():
                    metrics[f"{key}_{k}"] = v
            else:
                # metrics[key] = metric(estimate_result_t, y_t)
                dictionary = metric(estimate_result_t, y_t)
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
            "dataX": estimate_result_t,
            "dataY": y_t,
            "dataL": L,
            "name": batch["name"],
            "index": batch["index"],
            "label": batch["label"],
        }

    def configure_optimizers(self):
        # params = [p for p in self.model.parameters() if p.numel() > 0]
        # params = list(self.model.parameters())

        # if hasattr(self, "input_norm"):
        #     params += list(self.input_norm.parameters())

        # if hasattr(self, "special_loss"):
        #     params += list(self.special_loss.parameters())

        # if hasattr(self, "finalSMoother"):
        #     params += list(self.finalSMoother.parameters())

        # params = [p for p in params if p.numel() > 0]
        params = list(self.parameters())
        optimizer = optim.AdamW(params, lr=self.lr, fused=True)
        # check the tranning has overfit batch or not
        overfit_check = (
            hasattr(self.trainer, "overfit_batches") and self.trainer.overfit_batches
        )
        if overfit_check:
            # Disable EMA for fine-tuning
            if hasattr(self, "ema"):
                del self.ema

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                cooldown=0,
                min_lr=1e-8,
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

        accumulation = self.trainer.accumulate_grad_batches if self.trainer else 1
        num_batches = len(self.trainer.datamodule.train_dataloader())
        effected_batches = num_batches // self.trainer.world_size // accumulation

        if self.config["ema"]["enable"]:
            self.ema = diffusers.training_utils.EMAModel(
                parameters=params,
                decay=1 - 1 / (effected_batches),
                update_after_step=effected_batches,
                use_ema_warmup=True,
                inv_gamma=effected_batches,
                foreach=True,
            )
            self.ema.to(self.device)

        # Phase 1: Initial training with Cosine Warmup and EMA
        if self.config.get("train_phase", "initial") == "initial":

            scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=effected_batches * self.config["warm_up"],
                num_training_steps=(
                    self.trainer.max_steps
                    if self.trainer.max_steps > 0
                    else self.trainer.max_epochs * effected_batches
                ),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss_total/train_step",  # this doesn't matter for cosine scheduler
                    "interval": "step",
                    "frequency": accumulation,
                },
            }

        # Phase 2: Fine-tuning with ReduceLROnPlateau and no EMA
        elif self.config.get("train_phase") == "finetune":
            # # Disable EMA for fine-tuning
            # if hasattr(self, "ema"):
            #     del self.ema

            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=0.5,
                patience=3,
                cooldown=0,
                min_lr=1e-8,
                threshold=0.01,
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "metric_naive_distance_error_XY/val_mean",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(
                f"Invalid train_phase: {self.config.get('train_phase')}. Must be 'initial' or 'finetune'."
            )

    @overrides
    def _generate_metrics(self, suffix):
        metrics = {
            f"metric_naive_distance_error_XY/{suffix}": NaiveDistanceError(
                channel_index=[0, 1],
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
            f"metric_naive_distance_error_Z/{suffix}": NaiveDistanceError(
                channel_index=[2],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            f"metric_naive_distance_error/{suffix}": NaiveDistanceError(
                channel_index=[0, 1, 2],
                sampling_rate=self.config["sampling_rate"],
                avg_win=self.config["target_duration"],
            ),
            # acc
            #
            f"metric_acc_simVector/{suffix}": CosSimMetric(
                channel_index=[0, 1, 2], norm=False
            ),
            f"metric_acc_simVector_X/{suffix}": CosSimMetric(
                channel_index=[0], norm=False
            ),
            f"metric_acc_simVector_Y/{suffix}": CosSimMetric(
                channel_index=[1], norm=False
            ),
            f"metric_acc_simVector_Z/{suffix}": CosSimMetric(
                channel_index=[2], norm=False
            ),
            f"metric_acc_simVector_norm/{suffix}": CosSimMetric(
                channel_index=[0, 1, 2], norm=True
            ),
            # snr
            f"metric_acc_snr/{suffix}": mSNR(channel_index=[0, 1, 2], norm=False),
            f"metric_acc_snr_X/{suffix}": mSNR(channel_index=[0], norm=False),
            f"metric_acc_snr_Y/{suffix}": mSNR(channel_index=[1], norm=False),
            f"metric_acc_snr_Z/{suffix}": mSNR(channel_index=[2], norm=False),
            #
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
            #
            f"metric_gyr_simVector/{suffix}": CosSimMetric(
                channel_index=[3, 4, 5], norm=False
            ),
            f"metric_gyr_simVector_X/{suffix}": CosSimMetric(
                channel_index=[3], norm=False
            ),
            f"metric_gyr_simVector_Y/{suffix}": CosSimMetric(
                channel_index=[4], norm=False
            ),
            f"metric_gyr_simVector_Z/{suffix}": CosSimMetric(
                channel_index=[5], norm=False
            ),
            f"metric_gyr_simVector_norm/{suffix}": CosSimMetric(
                channel_index=[3, 4, 5], norm=True
            ),
            # snr
            f"metric_gyr_snr/{suffix}": mSNR(channel_index=[3, 4, 5], norm=False),
            f"metric_gyr_snr_X/{suffix}": mSNR(channel_index=[3], norm=False),
            f"metric_gyr_snr_Y/{suffix}": mSNR(channel_index=[4], norm=False),
            f"metric_gyr_snr_Z/{suffix}": mSNR(channel_index=[5], norm=False),
        }

        for key, metric in metrics.items():
            self.add_module(key, metric)
        return metrics