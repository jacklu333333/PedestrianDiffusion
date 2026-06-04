from .common_imports import *
from .utils import *


class mVAEModel(nn.Module):
    def __init__(self, latent_dim, eps, sample_size=32):
        super(mVAEModel, self).__init__()
        self.eps = eps

        self.model = diffusers.models.AutoencoderKL(
            in_channels=6,
            out_channels=6,
            down_block_types=(
                # "DownEncoderBlock2D",  # High res
                # "DownEncoderBlock2D",
                # "DownEncoderBlock2D",
                "AttnDownEncoderBlock2D",  # Low res (Bottleneck)
                # "DownEncoderBlock2D",
                # "DownEncoderBlock2D",
                # "DownEncoderBlock2D",
                # "DownEncoderBlock2D",
            ),
            up_block_types=(
                # "UpDecoderBlock2D",
                # "UpDecoderBlock2D",
                # "UpDecoderBlock2D",
                # "UpDecoderBlock2D",
                "AttnUpDecoderBlock2D",  # Low res (Bottleneck)
                # "UpDecoderBlock2D",
                # "UpDecoderBlock2D",
                # "UpDecoderBlock2D",  # High res
            ),
            block_out_channels=(
                32,
                # 64,
                # 128,
                # 256,
                # 256,
                # 128,
                # 512,
                # 512,
            ),
            latent_channels=latent_dim,
            layers_per_block=1,
            # act_fn="mish",
            sample_size=sample_size,
            # latent_channels=,
            # norm_num_groups=,
            # scaling_factor=,
            # shift_factor=,
            # latents_mean=,
            # latents_std=,
            # force_upcast=,
            # use_quant_conv=,
            # use_post_quant_conv=,
            # mid_block_add_attention=,
        )
        self.scale_factor = 1 / self.model.config.scaling_factor
        # learnable log variances for weighting each of the 3 loss components
        self.log_vars = nn.Parameter(torch.zeros(3))
        with torch.no_grad():
            dummy_input = torch.randn(1, 6, sample_size, sample_size)
            latent_dist = self.model.encode(dummy_input).latent_dist
            self._latent_space_shape = latent_dist.mean.shape[1:]

    # affine is True
    def forward(self, x):
        # Directly pass x to the model without normalization or affine transform
        recon_batch = self.model(x)
        recong_x_physical = recon_batch.sample

        latent_dist = self.model.encode(x).latent_dist
        mu, logvar = latent_dist.mean, latent_dist.logvar

        return recong_x_physical, x, mu, logvar

    def encode(self, x, scale=True):
        if scale:
            out = self.model.encode(x).latent_dist.sample() / self.scale_factor
        else:
            out = self.model.encode(x).latent_dist.sample()
        return out

    def decode(self, x, scale=True):
        # raise NotImplementedError("Decode method is not implemented yet.")
        if scale:
            out = self.model.decode(x * self.scale_factor).sample
        else:
            out = self.model.decode(x).sample
        return out
