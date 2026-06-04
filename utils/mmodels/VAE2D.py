import torch
import torch.nn as nn
import torch.nn.functional as F


class VAE2D(nn.Module):
    def __init__(self, in_channels=12, latent_dim=1024):
        super(VAE2D, self).__init__()

        # --- ENCODER (1 Layer) ---
        # Input: (B, 12, 16, 8)
        self.in_shape = (16, 8)
        self.kernel_size = (16, 8)
        self.stride = (16, 8)
        self.padding = (0, 0)
        self.hidden_channels = 1024

        self.enc_layer = nn.Conv2d(
            in_channels,
            self.hidden_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.gn_enc = nn.GroupNorm(32, self.hidden_channels)

        # Output shape calculation:
        # 16 -> 1
        # 8  -> 1
        # Flatten size = 1024 * 1 * 1 = 1024
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

        self.dec_layer = nn.ConvTranspose2d(
            self.hidden_channels,
            in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self._latent_space_shape = torch.Size([latent_dim, 1])
        self.latent_dim = latent_dim

    def reparameterize(self, mu, logvar):
        # std = torch.exp(0.5 * logvar)
        std = (
            1.0 + logvar + 0.5 * torch.square(logvar)
        )  # Additive variance stabilization
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # 1. Encode
        x_encod = F.relu6(self.gn_enc(self.enc_layer(x)))

        # 2. Flatten & Bottleneck
        x_flat = x_encod.view(x_encod.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        # Reshape to (B, 1024, 1)
        z_out = z.view(z.size(0), self.latent_dim, 1)

        # 3. Decode
        z_dec = F.relu6(self.fc_dec(z))
        z_dec = z_dec.view(z_dec.size(0), *self.out_shape)
        z_dec = self.gn_dec(z_dec)

        recon = self.dec_layer(z_dec)
        assert (
            recon.shape == x.shape
        ), f"Reconstructed shape {recon.shape} does not match input shape {x.shape}"
        return recon, z_out, mu, logvar

    def encode(self, x):
        # 1. Encode
        x_encod = F.relu6(self.gn_enc(self.enc_layer(x)))

        # 2. Flatten & Bottleneck
        x_flat = x_encod.view(x_encod.size(0), -1)
        mu = self.fc_mu(x_flat)
        logvar = self.fc_logvar(x_flat)
        z = self.reparameterize(mu, logvar)

        # Reshape to (B, 1024, 1)
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
