import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch import optim
from .PedestrianDiffusionTime import PedestrianDiffusionTime


class PedestrianDiffusionTimeSimple(PedestrianDiffusionTime):
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

        timesteps = torch.zeros(
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

    def configure_optimizers(self):
        params = list(self.parameters())
        optimizer = optim.AdamW(params, lr=self.lr, fused=True)
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
