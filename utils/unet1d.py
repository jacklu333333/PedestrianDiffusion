import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """A block of two 1D convolutions, each followed by BatchNorm and ReLU."""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling block: max pooling followed by a double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool1d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling block: up-convolution followed by a double convolution."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose1d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad x1 if its size is different from x2
        diff = x2.size()[2] - x1.size()[2]
        x1 = F.pad(x1, [diff // 2, diff - diff // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet1D(nn.Module):
    """A 3-layer 1D UNet for time-series data."""

    def __init__(self, in_channels, out_channels):
        super(UNet1D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.inc = DoubleConv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)
        self.outc = nn.Conv1d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Decoder
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Output layer
        logits = self.outc(x)
        return logits


class Discriminator1D(nn.Module):
    """A 1D PatchGAN discriminator for CycleGAN."""

    def __init__(self, in_channels):
        super(Discriminator1D, self).__init__()

        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block."""
            layers = [
                nn.Conv1d(in_filters, out_filters, kernel_size=4, stride=2, padding=1)
            ]
            if normalize:
                # Using InstanceNorm1d as is common in CycleGAN
                layers.append(nn.InstanceNorm1d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(in_channels, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0)),
            nn.Conv1d(512, 1, kernel_size=4, padding=1, bias=False)
        )

    def forward(self, x):
        """
        Input shape: (batch_size, channels, sequence_length)
        Output shape: (batch_size, 1, N) where N is the number of patches.
        """
        return self.model(x)


# Example Usage:
# model = UNet1D(in_channels=6, out_channels=6)
# test_input = torch.randn(16, 6, 1024) # (batch, channels, time)
# output = model(test_input)
# print(output.shape) # Should be (16, 6, 1024)
class BiLSTM(nn.Module):
    """A 3-layer bi-directional LSTM model."""

    def __init__(self, in_channels, out_channels, hidden_size=128, num_layers=3):
        super(BiLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )
        self.fc = nn.Linear(hidden_size * 2, out_channels)  # *2 for bidirectional

    def forward(self, x):
        """
        Input shape: (batch_size, channels, sequence_length)
        Output shape: (batch_size, out_channels, sequence_length)
        """
        # LSTM expects input of shape (batch_size, seq_len, input_size/channels)
        x = x.permute(0, 2, 1)

        # Initialize hidden and cell states
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device
        )  # *2 for bidirectional
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(
            x.device
        )  # *2 for bidirectional

        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))

        # Pass the output of each time step to the fully connected layer
        out = self.fc(out)

        # Permute back to (batch_size, channels, sequence_length)
        out = out.permute(0, 2, 1)
        return out
