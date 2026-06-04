import pytorch_lightning as pl
import torch
import torch.nn as nn
from diffusers.models import AutoencoderKLCogVideoX

# from ldm.models.autoencoder import AutoencoderKLCogVideoX
from ldm.modules.losses.lpips import LPIPS


class SpectrumVAE3D(pl.LightningModule):
    """
    A wrapper for the AutoencoderKLCogVideoX model to be used as a VAE.
    This class mimics the structure of other VAEs in the LDM framework.
    """

    def __init__(
        self,
        ddconfig,
        lossconfig,
        embed_dim,
        ckpt_path=None,
        ignore_keys=[],
        image_key="image",
        colorize_nlabels=None,
        monitor=None,
    ):
        super().__init__()
        self.image_key = image_key
        self.model = AutoencoderKLCogVideoX(ddconfig=ddconfig, embed_dim=embed_dim)
        self.perceptual_loss = LPIPS().eval()
        self.monitor = monitor

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        """
        Encodes the input tensor into a posterior distribution.
        """
        posterior = self.model.encode(x)
        return posterior

    def decode(self, z):
        """
        Decodes the latent tensor back to the image space.
        """
        dec = self.model.decode(z)
        return dec

    def forward(self, input, sample_posterior=True):
        """
        Full autoencoding pass.
        """
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def get_input(self, batch, k):
        """
        Retrieves the input tensor from a batch dictionary.
        """
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx):
        # This is a placeholder training step.
        # The actual implementation will depend on the main training script.
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)

        if optimizer_idx == 0:
            # train encoder+decoder
            aeloss, log_dict_ae = self.loss(
                inputs,
                reconstructions,
                posterior,
                0,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "aeloss",
                aeloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return aeloss

        if optimizer_idx == 1:
            # train the discriminator
            discloss, log_dict_disc = self.loss(
                inputs,
                reconstructions,
                posterior,
                1,
                self.global_step,
                last_layer=self.get_last_layer(),
                split="train",
            )
            self.log(
                "discloss",
                discloss,
                prog_bar=True,
                logger=True,
                on_step=True,
                on_epoch=True,
            )
            self.log_dict(
                log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False
            )
            return discloss

    def validation_step(self, batch, batch_idx):
        # This is a placeholder validation step.
        inputs = self.get_input(batch, self.image_key)
        reconstructions, posterior = self(inputs)
        aeloss, log_dict_ae = self.loss(
            inputs,
            reconstructions,
            posterior,
            0,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        discloss, log_dict_disc = self.loss(
            inputs,
            reconstructions,
            posterior,
            1,
            self.global_step,
            last_layer=self.get_last_layer(),
            split="val",
        )

        self.log("val/rec_loss", log_dict_ae["val/rec_loss"])
        self.log_dict(log_dict_ae)
        self.log_dict(log_dict_disc)
        return self.log_dict

    def configure_optimizers(self):
        # This is a placeholder optimizer configuration.
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(
            list(self.model.encoder.parameters())
            + list(self.model.decoder.parameters())
            + list(self.model.quant_conv.parameters())
            + list(self.model.post_quant_conv.parameters()),
            lr=lr,
            betas=(0.5, 0.9),
        )
        opt_disc = torch.optim.Adam(
            self.loss.discriminator.parameters(), lr=lr, betas=(0.5, 0.9)
        )
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        """
        Returns the last layer of the decoder.
        """
        return self.model.decoder.conv_out.weight

    @torch.no_grad()
    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            if x.shape[1] > 3:
                # colorize with random projection
                assert xrec.shape[1] > 3
                x = self.to_rgb(x)
                xrec = self.to_rgb(xrec)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log
