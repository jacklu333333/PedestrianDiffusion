import torch
import torch.nn as nn
from torch import optim

from diffusers.models import (
    UNet2DConditionModel,
    UNet3DConditionModel,
    UNetSpatioTemporalConditionModel,
)

from .common_imports import *
from .PedestrianDiffusion import PedestrianDiffusion
from .utils import *


class PedestrianDiffusionTime(PedestrianDiffusion):
    def __init__(self, config):
        super().__init__(config)
        del self.toObservation, self.deObservation
        self.toObservation = nn.Sequential(
            Rearrange("b c (t1 t2)-> b c 1 t1 t2", t1=10, t2=10),
        )
        self.deObservation = nn.Sequential(
            Rearrange("b c 1 t1 t2-> b c (t1 t2)", t1=10, t2=10),
        )

        del self.sensorProcessor, self.finalLabelProcessor, self.labelProcessor
        self.sensorProcessor = nn.Sequential()
        self.finalLabelProcessor = nn.Sequential()
        self.labelProcessor = nn.Sequential()

        del self.toXYZ
        self.toXYZ = nn.Sequential()

        del self.VAE
        self.VAE = VAEConv1D(
            input_dim=6,
            hidden_dim=self.config["latent_dim"],
            latent_dim=self.config["latent_dim"],
            seq_len=self.config["window_size"],
        )
        self.VAE.deleteDecoder()
        for param in self.VAE.parameters():
            param.requires_grad = False

        del self.model
        self.model = UNet3DConditionModel(
            sample_size=(10, 10),
            in_channels=12,
            out_channels=6,
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

        latent_space = self.VAE.encode(x).view(x.shape[0], 1, self.config["latent_dim"])
        encoding = torch.cat([latent_space, encoding], dim=1)
        attention_mask = torch.cat(
            [
                torch.ones(attention_mask.shape[0], 1, device=attention_mask.device),
                attention_mask,
            ],
            dim=1,
        )

        timesteps = torch.randint(
            0,
            self.config["target_time_steps"],
            (x.shape[0],),
            requires_grad=False,
            device=x.device,
        )

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

        target = y

        y_hat = estimate
        y_hat_t, target_t = self.postprocessing((y_hat, target, L))

        losses = self.compute_loss(
            freq_estimate=None,
            freq_target=None,
            #
            time_estimate=y_hat_t,
            time_target=target_t,
            #
            timesteps=timesteps,
            mode=mode,
        )

        masks = ~self.mask_generation(
            dataL=L,
            window_size=self.config["window_size"],
            batch_size=batch_size,
            channels=channels,
        )

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
        latent_space = self.VAE.encode(x).view(x.shape[0], 1, self.config["latent_dim"])
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
            estimate_result = self.scheduler.scale_model_input(estimate_result, t)

            estimate = self.forward_model(
                x=torch.cat([estimate_result, x], dim=1),
                original=None,
                t=t.unsqueeze(0).repeat(batch_size).to(self.device),
                class_labels=batch["datasets"],
                encoding=encoding,
                attention_mask=attention_mask,
            )

            # assert torch.isfinite(estimate).all()
            estimate_result = self.step_backward(
                original_input=estimate_result,
                estimate_noise=estimate,
                t=t.unsqueeze(0).repeat(batch_size),
                target_steps=target_testing_steps,
            )

        estimate_result_t, y_t = self.postprocessing((estimate_result, y, L))
        # estimate_result_t = self.finalSMoother(estimate_result_t)
        losses = self.compute_loss(
            freq_estimate=None,
            freq_target=None,
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

    def forward_model(
        self,
        x: torch.Tensor,
        original: torch.tensor,
        t: torch.Tensor,
        encoding: torch.tensor,
        attention_mask: torch.tensor,
        class_labels,
    ):
        assert torch.isfinite(
            encoding
        ).all(), 'The encoding is not finite in the function "forward_model"'

        out = self.toObservation(x)

        out = self.model(
            sample=out,
            timestep=t,
            encoder_hidden_states=encoding,
            attention_mask=attention_mask,
        ).sample
        assert torch.isfinite(
            out
        ).all(), 'The estimate is not finite in the function "forward_model"'

        out = self.deObservation(out)
        return out
