import torch
import torch.nn as nn
import torch.nn.functional as F


class VAEConv1D(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, seq_len=100):
        """
        Args:
            input_dim (int): Number of channels (features) in the input.
            hidden_dim (int): Number of filters in the convolutional layer (acts as hidden dimension).
            latent_dim (int): Dimension of the latent space (z).
            seq_len (int): Length of the time series (kernel size).
        """
        super(VAEConv1D, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # --- ENCODER ---
        # 1D Convolution
        # Input: (batch, input_dim, seq_len)
        # Kernel size matches seq_len to maximize observation window.
        self.encoder_conv = nn.Conv1d(
            in_channels=input_dim,
            out_channels=hidden_dim,
            kernel_size=seq_len,
            stride=1,
            padding=0,
        )

        # Linear projections for Mean (mu) and Log Variance (logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # --- DECODER ---
        # Project latent z back to hidden dimension for the decoder input
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        # 1D Transpose Convolution to reconstruct the sequence
        # We want to go from (batch, hidden_dim, 1) to (batch, input_dim, seq_len)
        self.decoder_conv = nn.ConvTranspose1d(
            in_channels=hidden_dim,
            out_channels=input_dim,
            kernel_size=seq_len,
            stride=1,
            padding=0,
        )

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: z = mu + std * epsilon
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, channel, timeseries)
        Returns:
            recon_x: Reconstructed data (batch, channel, timeseries)
            mu: Latent mean
            logvar: Latent log variance
        """
        # 1. Encoder
        # x shape: (batch, input_dim, seq_len)
        h = self.encoder_conv(x)  # (batch, hidden_dim, 1)

        # Flatten for linear layers
        h_flat = h.view(h.size(0), -1)  # (batch, hidden_dim)

        # 2. Latent Space
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # 3. Decoder
        # Project back to hidden_dim
        z_projected = self.decoder_input(z)  # (batch, hidden_dim)

        # Reshape for ConvTranspose1d
        z_reshaped = z_projected.view(
            z_projected.size(0), self.hidden_dim, 1
        )  # (batch, hidden_dim, 1)

        # Reconstruct
        recon_x = self.decoder_conv(z_reshaped)  # (batch, input_dim, seq_len)

        return recon_x, mu, logvar

    def encode(self, x):
        # 1. Encoder
        h = self.encoder_conv(x)
        h_flat = h.view(h.size(0), -1)

        # 2. Latent Space
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        z = self.reparameterize(mu, logvar)
        return z

    def decode(self, z):
        batch_size = z.size(0)

        # 1. Decoder
        z_projected = self.decoder_input(z)
        z_reshaped = z_projected.view(z_projected.size(0), self.hidden_dim, 1)

        # 2. Reconstruct
        recon_x = self.decoder_conv(z_reshaped)
        return recon_x

    def deleteDecoder(self):
        del self.decoder_conv
        del self.decoder_input

    def deleteEncoder(self):
        del self.encoder_conv
        del self.fc_mu
        del self.fc_logvar
