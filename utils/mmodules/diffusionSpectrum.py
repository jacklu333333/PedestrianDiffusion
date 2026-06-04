from .baseDiffusionModule import baseDiffusionModule
from .common_imports import *
from .utils import *


class diffusionSpectrum(baseDiffusionModule):
    def __init__(self, config):
        super(diffusionSpectrum, self).__init__(config)
        self.freq_loss = nn.L1Loss(reduction="sum")
        self.time_loss = nn.L1Loss(reduction="sum")
        self.time_sum_loss = nn.L1Loss(reduction="sum")
        self.freq_sim = nn.CosineSimilarity(dim=1)
        self.special_loss = spectrumMultiTaskLoss(dt=1.0 / self.config["sampling_rate"])
        self.eps = 1e-5

        # self.VAE = mVAEModel(self.config["latent_dim"], eps=1e-14)
        # for params in self.VAE.parameters():
        #     params.requires_grad = False

        self.model = UNet2DModel(
            sample_size=(32, 32),
            in_channels=12,
            out_channels=6,
            # center_input_sample: bool = False,
            # time_embedding_type: str = "positional",
            # time_embedding_dim: Optional[int] = None,
            # freq_shift: int = 0,
            # flip_sin_to_cos: bool = True,
            down_block_types=(
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
                "AttnDownBlock2D",
            ),
            mid_block_type=None,
            up_block_types=(
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
                "AttnUpBlock2D",
            ),
            # block_out_channels=(32, 448, 672, 896),
            # layers_per_block: int = 2,
            # mid_block_scale_factor: float = 1,
            # downsample_padding: int = 1,
            downsample_type="resnet",
            upsample_type="resnet",
            dropout=0.1,
            # act_fn: str = "silu",
            attention_head_dim=8,
            norm_num_groups=32,
            attn_norm_num_groups=32,
            # norm_eps: float = 1e-5,
            # resnet_time_scale_shift: str = "default",
            # add_attention: bool = True,
            # class_embed_type: Optional[str] = None,
            num_class_embeds=len(DATASET_DICT),
            num_train_timesteps=self.config["num_time_steps"],
        )

        # self.model = UNet2DConditionModel(
        #     sample_size=(32, 32),
        #     in_channels=12,
        #     out_channels=6,
        #     down_block_types=(
        #         # "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "CrossAttnDownBlock2D",
        #     ),
        #     up_block_types=(
        #         # "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "CrossAttnUpBlock2D",
        #     ),
        #     # center_input_sample: bool = False,
        #     # flip_sin_to_cos: bool = True,
        #     # freq_shift: int = 0,
        #     mid_block_type=None,
        #     # only_cross_attention=(True, True, True, True),
        #     # block_out_channels=(np.array([224, 448, 672, 896])).tolist(),
        #     block_out_channels=(
        #         (np.array([320, 640, 640, 1280, 1280], dtype=int) // 10 * 7)
        #     ).tolist(),
        #     # layers_per_block=1,
        #     # downsample_padding: int = 1,
        #     # mid_block_scale_factor: float = 1,
        #     dropout=0.1,
        #     # act_fn="mish",
        #     # norm_num_groups=32,
        #     # norm_eps: float = 1e-5,
        #     # cross_attention_dim=111,
        #     # transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        #     # reverse_transformer_layers_per_block: Optional[Tuple[Tuple[int]]] = None,
        #     encoder_hid_dim=self.config["latent_dim"],
        #     # encoder_hid_dim_type: Optional[str] = None,
        #     # attention_head_dim: Union[int, Tuple[int]] = 8,
        #     # num_attention_heads: Optional[Union[int, Tuple[int]]] = None,
        #     # dual_cross_attention=True,
        #     use_linear_projection=True,
        #     # class_embed_type: Optional[str] = None,
        #     # addition_embed_type: Optional[str] = None,
        #     # addition_time_embed_dim: Optional[int] = None,
        #     # num_class_embeds: Optional[int] = None,
        #     upcast_attention=True,
        #     # resnet_time_scale_shift="scale_shift",
        #     resnet_skip_time_act=False,
        #     # resnet_out_scale_factor=self.VAE.scale_factor,
        #     # time_embedding_type="fourier",
        #     # time_embedding_dim=256,
        #     # time_embedding_act_fn: Optional[str] = None,
        #     # timestep_post_act: Optional[str] = None,
        #     # time_cond_proj_dim: Optional[int] = None,
        #     # conv_in_kernel=3,
        #     # conv_out_kernel=3,
        #     # projection_class_embeddings_input_dim: Optional[int] = None,
        #     # attention_type: str = "default",
        #     # class_embeddings_concat: bool = False,
        #     # mid_block_only_cross_attention: Optional[bool] = None,
        #     # cross_attention_norm="group_norm",
        #     # addition_embed_type_num_heads: int = 64,
        # )

        del self.scheduler
        self.scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=self.config["num_time_steps"],
            beta_start=self.config["scheduler_s"],
            beta_end=self.config["scheduler_e"],
            beta_schedule=self.config["scheduler_mode"],
            # trained_betas=gen_linear_beta_t(
            #     num_timesteps=self.config["num_time_steps"]
            # ),
            solver_order=2,
            # prediction_type: str = "epsilon",
            # thresholding=True,
            # dynamic_thresholding_ratio: float = 0.995,
            # sample_max_value=1.0,
            # algorithm_type="dpmsolver++",
            # solver_type: str = "midpoint",
            # lower_order_final=True,
            euler_at_final=True,
            # use_karras_sigmas=True,
            # use_exponential_sigmas: Optional[bool] = False,
            # use_beta_sigmas=True,
            # use_lu_lambdas=True,
            final_sigmas_type="zero",  # "zero", "sigma_min"
            # lambda_min_clipped: float = -float("inf"),
            variance_type="v_prediction",
            # timestep_spacing: str = "linspace",
            # steps_offset: int = 0,
            # rescale_betas_zero_snr=True,
        )

        # self.scheduler = mDPMSolverMultistepScheduler(
        #     num_train_timesteps=self.config["num_time_steps"],
        #     beta_start=self.config["scheduler_s"],
        #     beta_end=self.config["scheduler_e"],
        #     beta_schedule=self.config["scheduler_mode"],
        #     # trained_betas=gen_linear_beta_t(
        #     #     num_timesteps=self.config["num_time_steps"]
        #     # ),
        #     solver_order=2,
        #     # prediction_type: str = "epsilon",
        #     # thresholding=True,
        #     # dynamic_thresholding_ratio: float = 0.995,
        #     # sample_max_value=1.0,
        #     algorithm_type="dpmsolver++",
        #     solver_type="midpoint",
        #     lower_order_final=True,
        #     euler_at_final=True,
        #     # use_karras_sigmas=True,
        #     # use_exponential_sigmas: Optional[bool] = False,
        #     # use_beta_sigmas=True,
        #     # use_lu_lambdas=True,
        #     final_sigmas_type="zero",  # "zero", "sigma_min"
        #     # lambda_min_clipped: float = -float("inf"),
        #     variance_type="learned_range",
        #     # timestep_spacing: str = "linspace",
        #     # steps_offset: int = 0,
        #     # rescale_betas_zero_snr=True,
        # )

        # self.scheduler = mDDPMScheduler(
        #     num_train_timesteps=self.config["num_time_steps"],
        #     beta_start=self.config["scheduler_s"],
        #     beta_end=self.config["scheduler_e"],
        #     beta_schedule=self.config["scheduler_mode"],
        #     # trained_betas: Optional[Union[np.ndarray, List[float]]] = None,
        #     # variance_type: str = "fixed_small",
        #     clip_sample=False,
        #     # prediction_type: str = "epsilon",
        #     # thresholding: bool = False,
        #     # dynamic_thresholding_ratio: float = 0.995,
        #     # clip_sample_range: float = 1.0,
        #     # sample_max_value: float = 1.0,
        #     timestep_spacing="linspace",
        #     # steps_offset: int = 0,
        #     # rescale_betas_zero_snr: bool = False,
        # )

    def on_train_start(self):
        return super().on_train_start()

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.config["ema"]["enable"] and hasattr(self, "ema"):
            params = [p for p in self.model.parameters() if p.numel() > 0]
            if hasattr(self, "input_norm"):
                params += [p for p in self.input_norm.parameters() if p.numel() > 0]
            self.ema.step(params)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        if (
            not self.config["peek_testing"]["enable"]
            or self.current_epoch < self.config["peek_testing"]["after"]
            or (self.current_epoch + 1 - self.config["peek_testing"]["after"])
            % self.config["peek_testing"]["n_epochs"]
            != 0
        ):
            return super().on_train_epoch_end()
        self.peeking()

        return super().on_train_epoch_end()

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
        class_labels,
    ):
        out = self.toObservation(x)

        out = self.model(
            sample=out,
            # hidden_states=x_,
            timestep=t,
            class_labels=class_labels,
            # encoder_hidden_states=encoding,
            # global_hidden_states=encoding,
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
        batch_size, channels = freq_target.shape[:2]
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
        aux_mask = (timesteps - 1).clamp(min=0) == 0
        # prim = time_sum_loss.clone()[aux_mask]
        # time_sum_loss = time_sum_loss.mean()

        # loss_aux = self.time_sum_loss(
        #     prim, torch.zeros_like(prim)
        # ) / aux_mask.sum().clamp(min=1)
        # # total_loss = freq_loss + time_loss + time_sum_loss
        # # total_loss = time_loss
        # # total_loss = freq_loss

        freq_sim = torch.vmap(self.freq_sim.forward)(
            freq_estimate.reshape(batch_size, channels, -1),
            freq_target.reshape(batch_size, channels, -1),
        )
        sign = torch.sign(freq_sim)
        freq_sim = (
            torch.pow(
                freq_sim.abs().clamp(min=0, max=1),
                timesteps.reshape(freq_sim.shape[0], 1, 1) + 1,
            )
            .mul(sign)
            .mean()
        )
        assert torch.isfinite(freq_sim).all(), "The frequency similarity is not finite"

        time_sim = torch.vmap(self.freq_sim.forward)(
            time_estimate.reshape(time_estimate.shape[0], channels, -1),
            time_target.reshape(time_target.shape[0], channels, -1),
        )
        sign = torch.sign(time_sim)
        time_sim = (
            torch.pow(
                time_sim.abs().clamp(min=0, max=1),
                timesteps.reshape(time_sim.shape[0], 1, 1) + 1,
            )
            .mul(sign)
            .mean()
        )
        assert torch.isfinite(time_sim).all(), "The time similarity is not finite"

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
        losses[f"freq_sim/{mode}"] = freq_sim
        losses[f"time_sim/{mode}"] = time_sim
        losses[f"freq_sim_c/{mode}"] = 1 - freq_sim
        losses[f"time_sim_c/{mode}"] = 1 - time_sim
        return losses

    def _generate_noise(self, x, y):
        noise = x
        # sample = x.reshape(x.shape[0], x.shape[1], 1, 1, -1)
        # print(cl.Fore.yellow, "y shape:", y.shape, cl.Style.reset)
        # print(
        #     cl.Fore.yellow,
        #     "Sample std:",
        #     sample.std(dim=-1, keepdim=True).shape,
        #     cl.Style.reset,
        # )
        # print(
        #     cl.Fore.yellow,
        #     "Sample mean:",
        #     sample.mean(dim=-1, keepdim=True).shape,
        #     cl.Style.reset,
        # )
        # noise = torch.randn_like(y) * sample.std(dim=-1, keepdim=True) + sample.mean(
        #     dim=-1, keepdim=True
        # )
        return noise
        # return torch.randn_like(y)
        # return torch.zeros_like(y)

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

        # latent_space = self.VAE.encode(self.toObservation(x))
        # encoding = latent_space.view(
        #     x.shape[0], self.config["latent_dim"], -1
        # ).swapaxes(-1, -2)

        # Classifier-Free Guidance: randomly drop conditioning
        if mode == "train" and self.config.get("cfg_drop_prob", 0.0) > 0:
            unconditional_encoding = torch.zeros_like(encoding)
            drop_mask = (
                torch.rand(batch_size, device=self.device)
                < self.config["cfg_drop_prob"]
            )
            encoding[drop_mask] = unconditional_encoding[drop_mask]

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

        batch_size, channels = x.shape[:2]
        x, y, _ = self.sample_noise((x, y), 0, dataL=L)

        # latent_space = self.VAE.encode(self.toObservation(x))
        # encoding = latent_space.view(
        #     x.shape[0], self.config["latent_dim"], -1
        # ).swapaxes(-1, -2)

        # Classifier-Free Guidance: randomly drop conditioning
        if mode == "train" and self.config.get("cfg_drop_prob", 0.0) > 0:
            unconditional_encoding = torch.zeros_like(encoding)
            drop_mask = (
                torch.rand(batch_size, device=self.device)
                < self.config["cfg_drop_prob"]
            )
            encoding[drop_mask] = unconditional_encoding[drop_mask]

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

        # y_hat = self.step_backward(
        #     original_input=pseudo_observation,
        #     estimate_noise=estimate,
        #     t=timesteps,
        #     target_steps=self.config["target_time_steps"],
        #     get_x0=False,
        #     different_timestamp=True,
        # )
        # # target = self._generate_target(y, noise, timesteps)
        # target = y
        y_hat_t, target_t = self.postprocessing((estimate, noise, L))

        losses = self.compute_loss(
            freq_estimate=estimate,
            freq_target=noise,
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

    def sequential_forward(self, batch, mode="test"):
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        encoding = batch["encoding"]
        batch_size, channels = x.shape[:2]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L, do_noise=False)
        """
        TODO: Implement the PipeLine approach for speed up
        """
        # pipeline = diffusers.DiffusionPipeline(
        # )

        target_testing_steps = (
            self.config["num_time_steps"]
            if mode == "test"
            else self.config["target_time_steps"]
        )
        timesteps = torch.arange(
            start=target_testing_steps - 1,
            end=-1,
            step=-1,
            device=x.device,
        )
        # latent_space = self.VAE.encode(self.toObservation(x))
        # encoding = latent_space.view(
        #     x.shape[0], self.config["latent_dim"], -1
        # ).swapaxes(-1, -2)
        self.scheduler.set_timesteps(target_testing_steps)
        unconditional_encoding = torch.zeros_like(encoding)
        cfg_scale = self.config.get("cfg_scale", 1.0)

        estimate_result = self._generate_noise(x, y)
        for t in timesteps:
            assert torch.isfinite(
                estimate_result
            ).all(), f"The estimate is not finite {t}"
            step_t = torch.ones(x.shape[0], dtype=torch.long, device=x.device) * t
            # Duplicate input for conditional and unconditional passes
            # estimate_result = torch.cat([estimate_result, x], dim=1)
            # model_input = torch.cat([estimate_result] * 2)
            # model_timestep = torch.cat([step_t] * 2)
            # model_encoding = torch.cat([unconditional_encoding, conditional_encoding])

            estimate = self.forward_model(
                # x=model_input,
                # original=None,
                # t=model_timestep,
                # encoding=model_encoding,
                x=torch.cat([estimate_result, x], dim=1),
                original=None,
                t=step_t,
                class_labels=batch["datasets"],
                encoding=encoding,
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
                t=step_t,
                target_steps=target_testing_steps,
                # get_x0=False if t > 0 else True,
                # get_x0=False,
                different_timestamp=False,
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
        params = list(self.model.parameters())

        if hasattr(self, "input_norm"):
            params += list(self.input_norm.parameters())

        if hasattr(self, "special_loss"):
            params += list(self.special_loss.parameters())

        if hasattr(self, "finalSMoother"):
            params += list(self.finalSMoother.parameters())

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

        # Phase 1: Initial training with Cosine Warmup and EMA
        if self.config.get("train_phase", "initial") == "initial":
            accumulation = self.trainer.accumulate_grad_batches if self.trainer else 1
            num_batches = len(self.trainer.datamodule.train_dataloader())
            valid_batches = num_batches // self.trainer.world_size // accumulation

            if self.config["ema"]["enable"]:
                self.ema = diffusers.training_utils.EMAModel(
                    parameters=params,
                    decay=1 - 1 / (valid_batches),
                    update_after_step=valid_batches,
                    use_ema_warmup=True,
                    inv_gamma=valid_batches,
                    foreach=True,
                )
                self.ema.to(self.device)

            scheduler = diffusers.optimization.get_cosine_schedule_with_warmup(
                optimizer=optimizer,
                num_warmup_steps=valid_batches * self.config["warm_up"],
                num_training_steps=(
                    self.trainer.max_steps
                    if self.trainer.max_steps > 0
                    else self.trainer.max_epochs * valid_batches
                ),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "loss_total/train",
                    "interval": "step",
                    "frequency": accumulation,
                },
            }

        # Phase 2: Fine-tuning with ReduceLROnPlateau and no EMA
        elif self.config.get("train_phase") == "finetune":
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
        else:
            raise ValueError(
                f"Invalid train_phase: {self.config.get('train_phase')}. Must be 'initial' or 'finetune'."
            )


# class diffusionTime(diffusionSpectrum):


# class cycleGan(baseDiffusionModule):
#     def __init__(self, config, *args, **kwargs):
#         super().__init__(config, *args, **kwargs)
#         self.generatorA = UNet3D(in_channels=6, out_channels=6)
#         self.generatorB = UNet3D(in_channels=6, out_channels=6)

#         self.discriminatorA = Discriminator(6, 64, 1)
#         self.discriminatorB = Discriminator(6, 64, 1)

#         self.loss = nn.MSELoss()
#         self.automatic_optimization = False

#     def configure_optimizers(self):

#         optimizerGA = optim.Adam(self.generatorA.parameters(), lr=self.lr)
#         optimizerGB = optim.Adam(self.generatorB.parameters(), lr=self.lr)
#         optimizerDA = optim.Adam(self.discriminatorA.parameters(), lr=self.lr)
#         optimizerDB = optim.Adam(self.discriminatorB.parameters(), lr=self.lr)

#         schedulerGA = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizerGA,
#             mode="min",
#             factor=0.9,
#             patience=10,
#             cooldown=5,
#             verbose=False,
#             min_lr=1e-8,
#         )
#         schedulerGB = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizerGB,
#             mode="min",
#             factor=0.9,
#             patience=10,
#             cooldown=5,
#             verbose=False,
#             min_lr=1e-8,
#         )
#         schedulerDA = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizerDA,
#             mode="min",
#             factor=0.9,
#             patience=10,
#             cooldown=5,
#             verbose=False,
#             min_lr=1e-8,
#         )
#         schedulerDB = optim.lr_scheduler.ReduceLROnPlateau(
#             optimizerDB,
#             mode="min",
#             factor=0.9,
#             patience=10,
#             cooldown=5,
#             verbose=False,
#             min_lr=1e-8,
#         )

#         return (
#             {
#                 "optimizer": optimizerGA,
#                 "lr_scheduler": {
#                     "scheduler": schedulerGA,
#                     "monitor": "loss/train_step",
#                     "interval": "step",
#                     "frequency": 1,
#                 },
#             },
#             {
#                 "optimizer": optimizerGB,
#                 "lr_scheduler": {
#                     "scheduler": schedulerGB,
#                     "monitor": "loss/train_step",
#                     "interval": "step",
#                     "frequency": 1,
#                 },
#             },
#             {
#                 "optimizer": optimizerDA,
#                 "lr_scheduler": {
#                     "scheduler": schedulerDA,
#                     "monitor": "loss/train_step",
#                     "interval": "step",
#                     "frequency": 1,
#                 },
#             },
#             {
#                 "optimizer": optimizerDB,
#                 "lr_scheduler": {
#                     "scheduler": schedulerDB,
#                     "monitor": "loss/train_step",
#                     "interval": "step",
#                     "frequency": 1,
#                 },
#             },
#         )

#     def training_step(self, batch, batch_idx):
#         x, y, L, classes = batch
#         x, y = self.sample_noise((x, y), 0, dataL=L)
#         batch_size, channel, freq, time, imaginary = x.shape

#         optimizerGA, optimizerGB, optimizerDA, optimizerDB = self.optimizers()
#         schedulerGA, schedulerGB, schedulerDA, schedulerDB = self.lr_schedulers()

#         # Train Generators
#         optimizerGA.zero_grad()
#         optimizerGB.zero_grad()

#         fake_y = self.generatorA(x)
#         fake_x = self.generatorB(y)

#         cycle_x = self.generatorB(fake_y)
#         cycle_y = self.generatorA(fake_x)

#         loss_cycle_x = F.mse_loss(cycle_x, x)
#         loss_cycle_y = F.mse_loss(cycle_y, y)

#         loss_GA = F.mse_loss(
#             self.discriminatorA(fake_y),
#             torch.ones((batch_size, 1), device=fake_y.device),
#         )
#         loss_GB = F.mse_loss(
#             self.discriminatorB(fake_x),
#             torch.ones((batch_size, 1), device=fake_x.device),
#         )

#         loss_G = loss_GA + loss_GB + 10 * (loss_cycle_x + loss_cycle_y)
#         self.manual_backward(loss_G)
#         torch.nn.utils.clip_grad_norm_(self.generatorA.parameters(), max_norm=1.0)
#         torch.nn.utils.clip_grad_norm_(self.generatorB.parameters(), max_norm=1.0)
#         optimizerGA.step()
#         optimizerGB.step()

#         # Train Discriminators
#         optimizerDA.zero_grad()
#         optimizerDB.zero_grad()

#         real_loss_A = F.mse_loss(
#             self.discriminatorA(y), torch.ones((batch_size, 1), device=y.device)
#         )
#         fake_loss_A = F.mse_loss(
#             self.discriminatorA(fake_y.detach()),
#             torch.zeros((batch_size, 1), device=fake_y.device),
#         )
#         loss_DA = (real_loss_A + fake_loss_A) / 2

#         real_loss_B = F.mse_loss(
#             self.discriminatorB(x), torch.ones((batch_size, 1), device=x.device)
#         )
#         fake_loss_B = F.mse_loss(
#             self.discriminatorB(fake_x.detach()),
#             torch.zeros((batch_size, 1), device=fake_x.device),
#         )
#         loss_DB = (real_loss_B + fake_loss_B) / 2

#         self.manual_backward(loss_DA)
#         self.manual_backward(loss_DB)
#         torch.nn.utils.clip_grad_norm_(self.discriminatorA.parameters(), max_norm=1.0)
#         torch.nn.utils.clip_grad_norm_(self.discriminatorB.parameters(), max_norm=1.0)
#         optimizerDA.step()
#         optimizerDB.step()

#         # Step the schedulers
#         schedulerGA.step(loss_GA)
#         schedulerGB.step(loss_GB)
#         schedulerDA.step(loss_DA)
#         schedulerDB.step(loss_DB)

#         # Log losses
#         self.log(
#             "loss_G",
#             loss_G,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#             sync_dist=True,
#         )
#         self.log(
#             "loss_DA",
#             loss_DA,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#             sync_dist=True,
#         )
#         self.log(
#             "loss_DB",
#             loss_DB,
#             on_step=True,
#             on_epoch=True,
#             prog_bar=True,
#             logger=True,
#             sync_dist=True,
#         )

#         return loss_G + loss_DA + loss_DB

#     def validation_step(self, batch, batch_idx):
#         x, y, L, classes = batch
#         x, y = self.sample_noise((x, y), 0, dataL=L, do_noise=False)

#         y_hat = self.generatorA(x)
#         y_hat, y = self.postprocessing((y_hat, y, L))

#         loss = self.loss(y_hat, y)
#         metrics = {}
#         for key, metric in self.metrics["val"].items():
#             metrics[key] = metric(y_hat, y)

#         self.log(
#             "loss/val",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             sync_dist=True,
#         )
#         self.log_dict(
#             metrics,
#             sync_dist=True,
#             on_step=False,
#             on_epoch=True,
#         )

#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y, L, classes = batch
#         x, y = self.sample_noise((x, y), 0, dataL=L, do_noise=False)

#         y_hat = self.generatorA(x)
#         y_hat, y = self.postprocessing((y_hat, y, L))

#         loss = self.loss(y_hat, y)

#         metrics = {}
#         for key, metric in self.metrics["test"].items():
#             metrics[key] = metric(y_hat, y)

#         self.log(
#             "loss/test",
#             loss,
#             on_step=False,
#             on_epoch=True,
#             sync_dist=True,
#             prog_bar=True,
#         )
#         self.log_dict(
#             metrics,
#             sync_dist=True,
#             on_step=False,
#             on_epoch=True,
#         )

#         return loss
