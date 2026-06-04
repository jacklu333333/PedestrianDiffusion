from .baseDiffusionModule import baseDiffusionModule
from .common_imports import *
from .utils import *


class DiffusionModelPL(baseDiffusionModule):
    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        # self.VAE = mVAEModle()
        # # load weight
        # # remove all key without the prefix model.
        # # weight = {k: v for k, v in weight.items() if "model." in k}
        # # replace all the prefix model. to VAE.
        # weight = {k.replace("VAE.", ""): v for k, v in weight.items()}
        # self.VAE.load_state_dict(weight, strict=True)
        # # freeze the VAE
        # for param in self.VAE.parameters():
        #     param.requires_grad = False
        # self.conditional_proj = nn.Sequential(
        #     nn.AdaptiveAvgPool1d(1),
        #     nn.Flatten(),
        # )
        # self.model = diffusers.models.UNet2DConditionModel(
        #     sample_size=(self.config["window_size"], 1),
        #     in_channels=6,
        #     out_channels=6,
        #     # center_input_sample=False,
        #     # flip_sin_to_cos=True,
        #     freq_shift=1.0,
        #     # down_block_types=(
        #     #     "CrossAttnDownBlock2D",
        #     #     "CrossAttnDownBlock2D",
        #     #     "CrossAttnDownBlock2D",
        #     #     "DownBlock2D",
        #     # ),
        #     # mid_block_type="UNetMidBlock2DCrossAttn",
        #     # up_block_types=(
        #     #     "UpBlock2D",
        #     #     "CrossAttnUpBlock2D",
        #     #     "CrossAttnUpBlock2D",
        #     #     "CrossAttnUpBlock2D",
        #     # ),
        #     # only_cross_attention=False,
        #     block_out_channels=(128, 128, 256, 256, 512, 512),
        #     down_block_types=(
        #         "DownBlock2D",
        #         "DownBlock2D",
        #         "DownBlock2D",
        #         "DownBlock2D",
        #         "CrossAttnDownBlock2D",
        #         "DownBlock2D",
        #     ),
        #     up_block_types=(
        #         "UpBlock2D",
        #         "CrossAttnUpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #         "UpBlock2D",
        #     ),
        #     # layers_per_block=2,
        #     # downsample_padding=1,
        #     # mid_block_scale_factor=1,
        #     dropout=0.1,
        #     # act_fn="mish",
        #     # norm_num_groups=32,
        #     # norm_eps=1e-5,
        #     cross_attention_dim=32,
        #     # transformer_layers_per_block=1,
        #     # reverse_transformer_layers_per_block=None,
        #     # encoder_hid_dim=None,
        #     # encoder_hid_dim_type=None,
        #     # attention_head_dim=8,
        #     # num_attention_heads=None,
        #     # dual_cross_attention=True,
        #     # use_linear_projection=False,
        #     # class_embed_type=None,
        #     # addition_embed_type=None,
        #     # addition_time_embed_dim=None,
        #     # num_class_embeds=None,
        #     upcast_attention=True,
        #     # resnet_time_scale_shift="default",
        #     # resnet_skip_time_act=False,
        #     # resnet_out_scale_factor=1.0,
        #     time_embedding_type="fourier",
        #     # time_embedding_dim=,
        #     time_embedding_act_fn="mish",
        #     timestep_post_act="mish",
        #     time_cond_proj_dim=512,
        #     # conv_in_kernel=3,
        #     # conv_out_kernel=3,
        #     # projection_class_embeddings_input_dim=512,
        #     # attention_type="default",
        #     # class_embeddings_concat=False,
        #     # mid_block_only_cross_attention=None,
        #     # cross_attention_norm=None,
        #     # addition_embed_type_num_heads=64,
        # )
        from .DiT import DiT

        self.model = DiT(
            input_size=32,
            patch_size=2,
            in_channels=2,
            hidden_size=1024,
            depth=24,
            num_heads=16,
            mlp_ratio=4.0,
            class_dropout_prob=0.1,
            num_classes=10,
            learn_sigma=True,
        )

        from .diffusion import create_diffusion

        self.diffusion = create_diffusion(
            timestep_respacing="",
            noise_schedule="squaredcos_cap_v2",
            # noise_schedule="linear",
            # use_kl=True,
            sigma_small=False,
            predict_xstart=True,
            learn_sigma=True,
            rescale_learned_sigmas=False,
            diffusion_steps=1000,
        )

        self.ema = ModelEmaV3(
            self.model,
            decay=config["ema_decay"],
        )

        self.loss = myspecialLoss(self.config["loss"])
        # self.loss = nn.HuberLoss(delta=0.0001)

    def on_train_start(self) -> None:
        return super().on_train_start()

    def forward(self, x):
        raise NotImplementedError

    def forward_model(self, x: torch.Tensor, original: torch.tensor, t: torch.Tensor):
        x = rearrange(x, "b c ( w h)-> b c w h", w=32, h=32)
        out = self.model(x)
        out = rearrange(out, "b c w h -> b c (w h)")

        return out

    def _get_regularization(self):
        l1_loss, l2_loss = torch.tensor(
            0.0, requires_grad=True, device=self.device
        ), torch.tensor(0.0, requires_grad=True, device=self.device)
        for param in self.model.parameters():
            l1_loss = l1_loss + torch.norm(param, 1)
            l2_loss = l2_loss + torch.norm(param, 2)

        return l1_loss, l2_loss

    def l1_l2_regularization(self):
        l1_loss, l2_loss = self._get_regularization()
        if not hasattr(self, "l1_loss"):
            self.l1_loss = l1_loss
            self.l2_loss = l2_loss

        return l1_loss / self.l1_loss, l2_loss / self.l2_loss

    def special_forward(self, batch, mode):
        x, y, L, activity = batch
        x, y = self.sample_noise((x, y), 0, dataL=L)
        timesteps = torch.randint(
            0,
            self.config["num_time_steps"],
            (x.shape[0],),
            requires_grad=False,
            device=x.device,
        )
        x, y = rearrange(x, "b c (w h) -> b c w h", w=32, h=32), rearrange(
            y, "b c (w h) -> b c w h", w=32, h=32
        )

        model_kwargs = dict(y=activity)
        loss_dict = self.diffusion.training_losses(
            self.model,
            y,
            timesteps,
            model_kwargs,
            noise=x,
        )
        loss = loss_dict["loss"].mean()
        l1_reg, l2_reg = self.l1_l2_regularization()

        self.log_dict(
            {
                f"total_loss/{mode}": loss + l1_reg + l2_reg,
                f"l1_loss/{mode}": l1_reg,
                f"l2_loss/{mode}": l2_reg,
                f"loss/{mode}": loss,
            },
            on_step=True if mode == "train" else False,
            on_epoch=True,
            prog_bar=True if mode == "train" else False,
            sync_dist=True,
        )

        return loss

    def training_step(self, batch, batch_idx):
        return self.special_forward(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.special_forward(batch, "val")

    def test_step(self, batch, batch_idx):
        # return self.sequential_testing(batch)
        x, y, L, activity = batch
        x, y = self.sample_noise((x, y), 0, dataL=L)

        x, y = rearrange(x, "b c (w h) -> b c w h", w=32, h=32), rearrange(
            y, "b c (w h) -> b c w h", w=32, h=32
        )

        model_kwargs = dict(y=activity)
        sample_fn = self.model.forward
        sample = self.diffusion.p_sample_loop(
            sample_fn,
            x.shape,
            x,
            clip_denoised=False,
            model_kwargs=model_kwargs,
            progress=False,
            device=self.device,
        )
        # loss = self.loss(sample, y)
        loss = F.mse_loss(sample, y)

        sample, y = rearrange(sample, "b c w h -> b c (w h)"), rearrange(
            y, "b c w h -> b c (w h)"
        )
        sample, y = self.postprocessing((sample, y, L))
        metrics = {}
        for key, metric in self.metrics["test"].items():
            metrics[key] = metric(sample, y)

        self.log("loss/test", loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log_dict(metrics, sync_dist=True, on_step=False, on_epoch=True)

        return loss

    def on_train_start(self):
        # for case in ["train", "val", "test"]:
        #     self.loss.clear_history(mode=case)
        return super().on_train_start()

    def on_train_batch_end(
        self,
        outputs,
        batch,
        batch_idx,
    ) -> None:
        if hasattr(self, "ema"):
            self.ema.update(self.model)
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def on_train_epoch_end(self):
        for case in ["train", "val", "test"]:
            self.loss.clear_history(mode=case)
        return super().on_train_epoch_end()

    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        sample = 10
        times = torch.randint(0, self.config["num_time_steps"], (sample,))
        seq = []
        x, y = batch
        seq.append(x)
        for t in reversed(range(1, self.config["num_time_steps"])):
            restoation, output, epsilon, total_loss = self.special_forward(
                (seq[-1], y), "pred"
            )
            assert torch.all(torch.isfinite(output)), "output is not finite"

            if t in times:
                seq.append(output)

        return seq

    def configure_optimizers(self):
        self.config["lr"] = self.lr
        self.save_hyperparameters(self.config)
        # optimizer = optim.SGD(self.parameters(), lr=self.lr)
        optimizer = optim.AdamW(self.parameters(), lr=self.lr)
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=self.lr)
        # optimizer = deepspeed.ops.adam.FusedAdam(self.parameters(), lr=self.lr)
        scheduler = CosineWarmupScheduler(optimizer, warmup=50, max_iters=250)
        # optimizer = optim.RMSprop(self.parameters(), lr=self.lr)
        # scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=2
        # )
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer,
        #     mode="min",
        #     factor=0.9,
        #     patience=10,
        #     cooldown=5,
        #     min_lr=1e-7,
        # )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "loss/train_step",
                "interval": "step",
                # "monitor": "loss/train_epoch",
                # "interval": "epoch",
                "frequency": 1,
            },
        }
