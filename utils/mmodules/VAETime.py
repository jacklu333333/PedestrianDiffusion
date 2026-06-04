from ..mloss import ControlVAELossTime
from ..mmodels import VAEConv1D, vaeLSTM
from .common_imports import *
from .utils import *
from .VAESpectrum import VAESpectrum


class VAETime(VAESpectrum):
    def __init__(self, config, target_modes: str = "x"):
        super(VAETime, self).__init__(config, target_modes=target_modes)
        self.sample_size = int(self.config["window_size"] ** 0.5)
        # self.VAE = ComplicatedVAEModel(
        #     self.config["latent_dim"], eps=1e-14, sample_size=self.sample_size
        # )
        # self.VAE = mVAEModel(self.config["latent_dim"], eps=1e-14, sample_size=16)

        # self.VAE = vaeLSTM(
        #     input_dim=6,
        #     hidden_dim=64,
        #     latent_dim=self.config["latent_dim"],
        #     num_layers=1,
        #     seq_len=self.config["window_size"],
        # )
        self.VAE = VAEConv1D(
            input_dim=6,
            hidden_dim=self.config["latent_dim"],
            latent_dim=self.config["latent_dim"],
            seq_len=self.config["window_size"],
        )

        self.loss_function = ControlVAELossTime(
            latent_space_numel=self.config["latent_dim"],
            target_kl=0.1,
        )

        if hasattr(self, "toObservation"):
            del self.toObservation
        if hasattr(self, "deObservation"):
            del self.deObservation
        if hasattr(self, "toYUV"):
            del self.toYUV
        if hasattr(self, "toXYZ"):
            del self.toXYZ
        self.toObservation = nn.Sequential(
            # batchNormalizeSensor(),
            # Rearrange("b c (w h) -> b c w h ", w=self.sample_size, h=self.sample_size),
            # Rearrange("b c (w h) -> b c w h ", w=self.config["window_size"], h=1),
        )
        self.deObservation = nn.Sequential(
            # Rearrange("b c w h -> b c (w h) ", w=self.sample_size, h=self.sample_size),
            # batchDenormalizeSensor(),
            # Rearrange("b c w h -> b c (w h) ", w=self.config["window_size"], h=1),
        )

        self.toYUV = nn.Sequential()
        self.toXYZ = nn.Sequential()

    def _samplewise_loss_components(self, x_hat, x, mu, logvar):
        """Compute per-sample recon/KL/total values for stats logging only."""
        batch_size = x_hat.shape[0]
        recon_acc = F.mse_loss(
            x_hat[:, :3] * 1000,
            x[:, :3] * 1000,
            reduction="none",
        ).view(batch_size, -1)
        recon_gyr = F.mse_loss(
            x_hat[:, 3:] * 1800 / torch.pi,
            x[:, 3:] * 1800 / torch.pi,
            reduction="none",
        ).view(batch_size, -1)

        recon_loss_acc = recon_acc.mean(dim=1)
        recon_loss_gyr = recon_gyr.mean(dim=1)
        recon_loss = recon_loss_acc + recon_loss_gyr
        kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        kl_div = kl_div / self.config["latent_dim"]
        total_loss = recon_loss + kl_div

        return {
            "recon_loss_acc": recon_loss_acc,
            "recon_loss_gyr": recon_loss_gyr,
            "recon_loss": recon_loss,
            "kl_div": kl_div,
            "total_loss": total_loss,
        }

    def _mean_std(self, values):
        mean_value = values.mean()
        std_value = values.std(unbiased=False)
        return mean_value, std_value

    @overrides
    def forward_model(self, x):
        x = self.toObservation(x)
        x_hat, mu, logvar = self.VAE(x)
        # recon_loss, kl_loss = self.loss_compute(x_hat, x, mu, logvar)
        # loss_dict = self.loss_function(x_hat, x, mu, logvar)
        # return x_hat, x, mu, logvar
        return x_hat, x, mu, logvar

    @overrides
    def regular_forward(self, batch, mode):
        x = batch["dataX"]
        y = batch["dataY"]
        L = batch["dataL"]
        label = batch["label"]

        x, y, _ = self.sample_noise((x, y), 0, dataL=L)

        if self.target_mode == "mix":
            x_hat, x, x_mu, x_logvar = self.forward_model(x)
            y_hat, y, y_mu, y_logvar = self.forward_model(y)

            # print(cl.Fore.red, x_hat.shape, x.shape, L.shape, cl.Style.reset)

            # x_hat, x = self.postprocessing((x_hat, x, L))
            # y_hat, y = self.postprocessing((y_hat, y, L))
            mse = F.mse_loss(x_hat, x) + F.mse_loss(y_hat, y)

            loss_dict_x = self.loss_function(
                x_hat,
                x,
                x_mu,
                x_logvar,
            )
            loss_dict_y = self.loss_function(
                y_hat,
                y,
                y_mu,
                y_logvar,
            )
            loss_dict = {
                key: loss_dict_x[key] + loss_dict_y[key] for key in loss_dict_x.keys()
            }

        else:
            if self.target_mode == "y":
                x, y = y, x
            elif self.target_mode == "x":
                x, y = x, y
            else:
                raise ValueError("The target mode is not defined")
            x_hat, _, mu, logvar = self.forward_model(x)
            loss_dict = self.loss_function(x_hat, x, mu, logvar)

            samplewise_losses = self._samplewise_loss_components(x_hat, x, mu, logvar)
            recon_acc_mean, recon_acc_std = self._mean_std(
                samplewise_losses["recon_loss_acc"]
            )
            recon_gyr_mean, recon_gyr_std = self._mean_std(
                samplewise_losses["recon_loss_gyr"]
            )
            recon_mean, recon_std = self._mean_std(samplewise_losses["recon_loss"])
            kl_mean, kl_std = self._mean_std(samplewise_losses["kl_div"])
            total_mean, total_std = self._mean_std(samplewise_losses["total_loss"])

            self.log_dict(
                {
                    f"recon_loss_acc_mean/{mode}": recon_acc_mean,
                    f"recon_loss_acc_std/{mode}": recon_acc_std,
                    f"recon_loss_gyr_mean/{mode}": recon_gyr_mean,
                    f"recon_loss_gyr_std/{mode}": recon_gyr_std,
                    f"recon_loss_mean/{mode}": recon_mean,
                    f"recon_loss_std/{mode}": recon_std,
                    f"kl_div_mean/{mode}": kl_mean,
                    f"kl_div_std/{mode}": kl_std,
                    f"total_loss_mean/{mode}": total_mean,
                    f"total_loss_std/{mode}": total_std,
                },
                on_step=True if mode == "train" else False,
                on_epoch=False if mode == "train" else True,
                prog_bar=False,
                sync_dist=True,
            )

            mse = F.mse_loss(x_hat, x)
            x_hat_t, x_t = self.postprocessing(
                (self.deObservation(x_hat), self.deObservation(x), L)
            )

            # if mode == "test":
            if True:
                unique_label = torch.unique(label)
                for i in unique_label:
                    idx = label == i
                    dataset_name = INV_DATASET_DICT[i.item()]

                    i_losses = {
                        key: value[idx] for key, value in samplewise_losses.items()
                    }
                    i_recon_acc_mean, i_recon_acc_std = self._mean_std(
                        i_losses["recon_loss_acc"]
                    )
                    i_recon_gyr_mean, i_recon_gyr_std = self._mean_std(
                        i_losses["recon_loss_gyr"]
                    )
                    i_recon_mean, i_recon_std = self._mean_std(i_losses["recon_loss"])
                    i_kl_mean, i_kl_std = self._mean_std(i_losses["kl_div"])
                    i_total_mean, i_total_std = self._mean_std(i_losses["total_loss"])

                    self.log_dict(
                        {
                            f"recon_loss_{dataset_name}/{mode}": i_recon_mean,
                            f"recon_loss_acc_{dataset_name}_mean/{mode}": i_recon_acc_mean,
                            f"recon_loss_acc_{dataset_name}_std/{mode}": i_recon_acc_std,
                            f"recon_loss_gyr_{dataset_name}_mean/{mode}": i_recon_gyr_mean,
                            f"recon_loss_gyr_{dataset_name}_std/{mode}": i_recon_gyr_std,
                            f"kl_div_{dataset_name}/{mode}": i_kl_mean,
                            f"recon_loss_{dataset_name}_mean/{mode}": i_recon_mean,
                            f"recon_loss_{dataset_name}_std/{mode}": i_recon_std,
                            f"kl_div_{dataset_name}_mean/{mode}": i_kl_mean,
                            f"kl_div_{dataset_name}_std/{mode}": i_kl_std,
                            f"total_loss_{dataset_name}_mean/{mode}": i_total_mean,
                            f"total_loss_{dataset_name}_std/{mode}": i_total_std,
                        },
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
                    metrics = {}
                    for key, metric in self.metrics[mode].items():
                        dictionary = metric(x_hat_t[idx], x_t[idx])
                        for k, v in dictionary.items():
                            # insert {INV_DATASET_DICT[i.item()]} to key befor /
                            key_parts = key.split("/", 1)
                            key_parts[0] = (
                                f"{key_parts[0]}_{INV_DATASET_DICT[i.item()]}"
                            )
                            new_key = "/".join(key_parts) + f"_{k}"
                            metrics[new_key] = v

                    self.log_dict(
                        metrics,
                        on_step=False,
                        on_epoch=True,
                        prog_bar=False,
                        sync_dist=True,
                    )
            # x_hat, x = self.postprocessing((x_hat, x, L))
        # psnr = self.PSNR(x_hat, x)
        # ssim = self.SSIM(x_hat, x)
        # self.log_dict(
        #     {
        #         f"psnr/{mode}": psnr,
        #         f"ssim/{mode}": ssim,
        #     },
        #     on_step=True if mode == "train" else False,
        #     on_epoch=False if mode == "train" else True,
        #     prog_bar=True,
        #     sync_dist=True,
        # )

        self.log(
            f"mse/{mode}",
            mse,
            on_step=True if mode == "train" else False,
            on_epoch=False if mode == "train" else True,
            prog_bar=True,
            sync_dist=True,
        )
        losses = {}
        for key, value in loss_dict.items():
            losses[f"{key}/{mode}"] = value
        self.log_dict(
            losses,
            on_step=True if mode == "train" else False,
            on_epoch=False if mode == "train" else True,
            prog_bar=True,
            sync_dist=True,
        )
        result_dict = {}
        if mode == "test":
            encoded = self.VAE.encode(x)
            result_dict["encoded"] = encoded
            # result_dict["label"] = batch["label"]

        for key, value in batch.items():
            result_dict[key] = value

        return (
            loss_dict["recon_loss_acc"] + loss_dict["recon_loss_gyr"],
            loss_dict["kl_loss"],
            loss_dict["loss_total"],
            mse,
        ), result_dict
