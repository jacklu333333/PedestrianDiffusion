import torch
import torch.nn as nn
import torch.nn.functional as F


class vaeLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, seq_len=100):
        """
        Args:
            input_dim (int): Number of channels (features) in the input.
            hidden_dim (int): Hidden size of the LSTM.
            latent_dim (int): Dimension of the latent space (z).
            num_layers (int): Number of LSTM layers.
            seq_len (int): Length of the time series (required for the decoder).
        """
        super(vaeLSTM, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # --- ENCODER ---
        # Bidirectional LSTM
        # Input: (batch, seq_len, input_dim)
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        # Linear projections for Mean (mu) and Log Variance (logvar)
        # Note: Input is hidden_dim * 2 because the LSTM is bidirectional
        self.fc_mu = nn.Linear(hidden_dim * 2, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim * 2, latent_dim)

        # --- DECODER ---
        # Project latent z back to hidden dimension for the decoder input
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)

        # Decoder LSTM (Unidirectional is standard for reconstruction/generation)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_dim,  # We feed the projected latent vector
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False,  # Decoder is usually unidirectional
        )

        # Final projection to reconstruct original input dimension
        self.final_layer = nn.Linear(hidden_dim, input_dim)

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
        # 1. Permute to (batch, timeseries, channel) for LSTM
        # x input: [Batch, Channel, Time] -> [Batch, Time, Channel]
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)

        # 2. Encoder
        # output shape: (batch, seq_len, hidden_dim * 2)
        # hidden shape: (num_layers * 2, batch, hidden_dim)
        _, (h_n, _) = self.encoder_lstm(x)

        # Concatenate the last hidden state of forward and backward directions
        # h_n shape: [num_layers * num_directions, batch, hidden_size]
        # We take the last layer's forward and backward states
        # View assumes num_layers=1. For num_layers > 1, specific indexing is safer.
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_encoded = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim * 2)

        # 3. Latent Space
        mu = self.fc_mu(h_encoded)
        logvar = self.fc_logvar(h_encoded)
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        # 4. Decoder Preparation
        # Expand z to match sequence length to feed into Decoder LSTM
        # Method: Repeat z across time steps
        z_projected = self.decoder_input(z)  # (batch, hidden_dim)

        # Reshape to (batch, seq_len, hidden_dim)
        z_repeated = z_projected.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 5. Decoder
        # decoder_output shape: (batch, seq_len, hidden_dim)
        decoder_output, _ = self.decoder_lstm(z_repeated)

        # Map back to original input dimension
        recon_x = self.final_layer(decoder_output)  # (batch, seq_len, input_dim)

        # 6. Permute back to (batch, channel, timeseries)
        recon_x = recon_x.permute(0, 2, 1)

        return recon_x, mu, logvar

    def encode(self, x):
        # 1. Permute to (batch, timeseries, channel) for LSTM
        x = x.permute(0, 2, 1)

        # 2. Encoder
        _, (h_n, _) = self.encoder_lstm(x)

        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_encoded = torch.cat([h_forward, h_backward], dim=1)  # (batch, hidden_dim * 2)

        # 3. Latent Space
        mu = self.fc_mu(h_encoded)
        logvar = self.fc_logvar(h_encoded)
        z = self.reparameterize(mu, logvar)  # (batch, latent_dim)

        return z

    def decode(self, z):
        batch_size = z.size(0)

        # 1. Decoder Preparation
        z_projected = self.decoder_input(z)  # (batch, hidden_dim)

        # Reshape to (batch, seq_len, hidden_dim)
        z_repeated = z_projected.unsqueeze(1).repeat(1, self.seq_len, 1)

        # 2. Decoder
        decoder_output, _ = self.decoder_lstm(z_repeated)

        # Map back to original input dimension
        recon_x = self.final_layer(decoder_output)  # (batch, seq_len, input_dim)

        # 3. Permute back to (batch, channel, timeseries)
        recon_x = recon_x.permute(0, 2, 1)

        return recon_x

    def deleteDecoder(self):
        del self.decoder_lstm
        del self.final_layer
        del self.decoder_input

    def deleteEncoder(self):
        del self.encoder_lstm
        del self.fc_mu
        del self.fc_logvar
