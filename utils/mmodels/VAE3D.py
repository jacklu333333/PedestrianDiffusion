import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE3D_1024(nn.Module):
    def __init__(self, in_channels=1, latent_dim=1024):
        super(VAE3D_1024, self).__init__()

        # --- ENCODER (1 Layer) ---
        # Input: (B, 1, 16, 32, 2)
        self.in_shape = (16, 8, 2)
        # self.kernel_size = (2, 2, 1)
        # self.stride = (2, 2, 1)
        self.kernel_size = (16, 8, 2)
        self.stride = (16, 8, 2)
        self.padding = (0, 0, 0)
        self.hidden_channels = 1024
        # We perform one aggressive downsampling step
        self.enc_layer = nn.Conv3d(
            in_channels,
            self.hidden_channels,  # Jump straight to 128 features
            kernel_size=self.kernel_size,  # Keep depth '2' safe
            stride=self.stride,  # Downsample H and W by half
            padding=self.padding,  # Standard padding
        )
        self.gn_enc = nn.GroupNorm(32, self.hidden_channels)

        # Output shape calculation:
        # 16 -> 8
        # 32 -> 16
        # 2  -> 2 (Unchanged)
        # Flatten size = 128 * 8 * 16 * 2 = 32,768
        with torch.no_grad():
            dummy_input = torch.zeros(1, in_channels, *self.in_shape)
            dummy_output = self.enc_layer(dummy_input)
            self.out_shape = dummy_output.shape[1:]
            self.flat_dim = dummy_output.flatten(1).shape[1]

        # Latent Space
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flat_dim, latent_dim)

        # --- DECODER (1 Layer) ---
        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.gn_dec = nn.GroupNorm(32, self.hidden_channels)
        # We perform one aggressive upsampling step
        self.dec_layer = nn.ConvTranspose3d(
            self.hidden_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self._latent_space_shape = torch.Size([latent_dim, 1])
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        # 1. Encode
        x_encod = F.silu(self.gn_enc(self.enc_layer(x)))

        # 2. Flatten & Bottleneck
        x_flat = x_encod.view(x_encod.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        # Reshape to your requested (B, 1024, 1)
        z_out = z.view(z.size(0), self.latent_dim, 1)

        # 3. Decode
        z = F.silu(self.fc_dec(z))
        # z = z.view(z.size(0), 128, 8, 4, 2)  # Unflatten
        z = z.view(z.size(0), *self.out_shape)  # Unflatten
        z = self.gn_dec(z)

        recon = self.dec_layer(z)
        assert (
            recon.shape == x.shape
        ), f"Reconstructed shape {recon.shape} does not match input shape {x_encod.shape}"
        return recon, z_out, mu, logvar

    def encode(self, x):
        # 1. Encode
        x = F.silu(self.gn_enc(self.enc_layer(x)))

        # 2. Flatten & Bottleneck
        x_flat = x.view(x.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        # Reshape to your requested (B, 1024, 1)
        z_out = z.view(z.size(0), self.latent_dim, 1)
        return z_out

    def deleteDecoder(self):
        del self.dec_layer
        del self.fc_dec
        del self.gn_dec

    def deleteEncoder(self):
        del self.enc_layer
        del self.fc_mu
        del self.fc_logvar
        del self.gn_enc


# # --- Verification ---
# model = ShallowVAE(latent_dim=1024)
# x = torch.randn(10, 1, 16, 32, 2)
# recon, latent, _, _ = model(x)

# print(f"Input: {x.shape}")
# print(f"Recon: {recon.shape}")
# print(f"Latent: {latent.shape}")
