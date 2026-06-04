import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class stempsEmbedding(nn.Module):
    def __init__(self, timesteps, in_channels):
        super(stempsEmbedding, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=timesteps, embedding_dim=in_channels
        )

    def forward(self, x, t):
        time_emb = self.embedding(t)
        time_emb = time_emb.view(-1, x.shape[1], 1, 1, 1)

        return x + time_emb


class PositionEncoding3D(nn.Module):
    def __init__(self, channels, depth, height, width):
        super(PositionEncoding3D, self).__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.channels = channels

        self.positional_encoding = self.create_positional_encoding(
            channels, depth, height, width
        )

    def create_positional_encoding(self, channels, depth, height, width):
        total = depth * height * width
        position = torch.arange(total).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, channels, 2) * (-math.log(10000.0) / channels)
        )
        pe = torch.zeros(total, 1, channels)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        pe = rearrange(
            pe, "(d h w) b c -> b c d h w", c=channels, d=depth, h=height, w=width
        )
        return pe

    def forward(self, x):
        import colored as cl

        # print(cl.Fore.red + f"x shape: {x.shape}" + cl.Style.reset)
        # print(
        #     cl.Fore.red
        #     + f"pe shape: {self.positional_encoding.to(x.device)[:, : x.shape[1], : x.shape[2], : x.shape[3], : x.shape[4]].shape}"
        #     + cl.Style.reset
        # )
        return x + self.positional_encoding.to(x.device)[
            :, : x.shape[1], : x.shape[2], : x.shape[3], : x.shape[4]
        ].expand_as(x)


class ResNetBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, time_steps=1000):
        super(ResNetBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm3d(out_channels)
        self.dropout = nn.Dropout3d(0.1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm3d(out_channels)
        self.activation = nn.Mish(inplace=True)
        # self.position_encoding = PositionEncoding3D(
        #     channels=in_channels, depth=64, height=64, width=64
        # )
        # self.time_embedding = stempsEmbedding(time_steps, out_channels)
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x, t):
        # out1 = self.position_encoding(x)
        out1 = self.conv1(x)
        out1 = self.norm1(out1)
        out1 = self.activation(out1)
        out1 = self.dropout(out1)
        # out1 = self.time_embedding(out1, t)
        out2 = self.conv2(out1)
        out2 = self.norm2(out2)
        x = self.residual_conv(x)
        out2 = self.activation(out2)
        return out2


class Attention3D(nn.Module):
    def __init__(self, channels, num_steps, cross_dim=None):
        super(Attention3D, self).__init__()
        self.cross_dim = cross_dim
        self.query = nn.Conv3d(channels, max(1, channels // 8), kernel_size=1)
        self.key = nn.Conv3d(channels, max(1, channels // 8), kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        if cross_dim is not None:
            self.cross_key = nn.Conv3d(cross_dim, max(1, channels // 8), kernel_size=1)
            self.cross_value = nn.Conv3d(cross_dim, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.time_embedding = stempsEmbedding(
            num_steps, cross_dim if cross_dim else channels
        )
        self.position_encoding = PositionEncoding3D(
            channels=cross_dim if cross_dim else channels, depth=64, height=64, width=64
        )

    def forward(self, x, t, hidden_state=None):
        batch_size, C, Freq, T, I = x.size()
        query = rearrange(self.query(x), "b c f t i -> b (f t i) c")
        if self.cross_dim is not None and hidden_state is not None:
            key = self.time_embedding(hidden_state, t)
            key = rearrange(self.cross_key(key), "b c f t i -> b c (f t i)")

            value = self.position_encoding(hidden_state)
            value = rearrange(self.cross_value(value), "b c f t i -> b c (f t i)")
        else:
            key = self.time_embedding(x, t)
            key = rearrange(self.key(key), "b c f t i -> b c (f t i)")

            value = self.position_encoding(x)
            value = rearrange(self.value(value), "b c f t i -> b c (f t i)")

        attention = torch.bmm(query, key)
        # normalize with value
        # attention = attention / attention.max(dim=-1, keepdim=True)[0]
        attention = F.softmax(attention, dim=-1)
        # replace NaN with
        out = torch.bmm(value, rearrange(attention, "b fti c -> b c fti"))
        out = rearrange(out, "b c (f t i) -> b c f t i", f=Freq, t=T, i=I)
        out = self.gamma * out + x
        return out


class UNet3DTimestapHidden(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        skip_connections=True,
        layers=3,
        num_steps=1000,
        cross_dim=None,
    ):
        super(UNet3DTimestapHidden, self).__init__()
        self.downBlocks = nn.ModuleList()
        self.upBlocks = nn.ModuleList()
        self.attentionBlocks = nn.ModuleList()
        self.skip_connections = skip_connections
        self.cross_dim = cross_dim

        in_dim = in_channels
        hidden_dim = 64

        # Downsampling layers
        for _ in range(layers):
            self.downBlocks.append(
                nn.ModuleList(
                    [
                        ResNetBlock3D(in_dim, hidden_dim),
                        nn.Conv3d(
                            hidden_dim,
                            hidden_dim,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=(0, 0, 1),
                        ),
                        # (
                        #     nn.BatchNorm3d(hidden_dim)
                        #     if _ > 1
                        #     else nn.InstanceNorm3d(hidden_dim)
                        # ),
                        nn.InstanceNorm3d(hidden_dim),
                        nn.Mish(inplace=True),
                    ]
                )
            )
            self.attentionBlocks.append(
                Attention3D(hidden_dim, num_steps=num_steps, cross_dim=cross_dim)
            )
            in_dim = hidden_dim
            hidden_dim *= 2

        # Middle layer
        self.midLayer = ResNetBlock3D(in_dim, in_dim)

        # Upsampling layers
        for _ in range(layers):
            hidden_dim //= 2
            out = hidden_dim if _ < (layers - 1) else out_channels
            self.upBlocks.append(
                nn.ModuleList(
                    [
                        (
                            ResNetBlock3D(in_dim + hidden_dim, out)
                            if skip_connections
                            else ResNetBlock3D(in_dim, out)
                        ),
                        nn.ConvTranspose3d(
                            out,
                            out,
                            kernel_size=(3, 3, 3),
                            stride=(1, 1, 1),
                            padding=(0, 0, 1),
                        ),
                        # nn.BatchNorm3d(out) if _ < 1 else nn.InstanceNorm3d(out),
                        nn.InstanceNorm3d(out),
                        nn.Mish(inplace=True),
                    ]
                )
            )
            self.attentionBlocks.append(
                Attention3D(out, num_steps=num_steps, cross_dim=cross_dim)
            )
            in_dim = hidden_dim

        # self.position_encoding = PositionEncoding3D(
        #     channels=in_channels, depth=64, height=64, width=64
        # )
        # self.time_embedding = nn.Embedding(
        #     num_embeddings=num_steps, embedding_dim=in_channels
        # )

    def forward(self, x, t, hidden_state=None):
        original_shape = x.shape
        # x = self.position_encoding(x)
        # time_emb = self.time_embedding(t).view(-1, x.shape[1], 1, 1, 1)
        # print(f"Time embedding shape: {time_emb.shape}")
        # x = x + time_emb

        if self.skip_connections:
            skip_connections = []

        for block, attn in zip(
            self.downBlocks, self.attentionBlocks[: len(self.downBlocks)]
        ):
            for layer in block:
                x = layer(x, t) if isinstance(layer, ResNetBlock3D) else layer(x)
            x = attn(x, t=t, hidden_state=hidden_state)
            if self.skip_connections:
                skip_connections.append(x)

        x = self.midLayer(x, t)
        if self.skip_connections:
            for block, attn in zip(
                self.upBlocks, self.attentionBlocks[len(self.downBlocks) :]
            ):
                sk = skip_connections.pop()
                x = torch.cat(
                    (
                        x[:, :, : sk.shape[2], : sk.shape[3], : sk.shape[4]],
                        sk,
                    ),
                    dim=1,
                )
                for layer in block:
                    x = layer(x, t) if isinstance(layer, ResNetBlock3D) else layer(x)
                x = attn(x, t=t, hidden_state=hidden_state)
        else:
            for block, attn in zip(
                self.upBlocks, self.attentionBlocks[len(self.downBlocks) :]
            ):
                for layer in block:
                    x = layer(x, t) if isinstance(layer, ResNetBlock3D) else layer(x)
                x = attn(x, t=t, hidden_state=hidden_state)

        x = x[:, :, : original_shape[2], : original_shape[3], : original_shape[4]]
        return x

    def encode(self, x, t):
        original_shape = x.shape
        for block, attn in zip(
            self.downBlocks, self.attentionBlocks[: len(self.downBlocks)]
        ):
            for layer in block:
                x = layer(x, t) if isinstance(layer, ResNetBlock3D) else layer(x)

        return nn.Flatten()(x)


def main():
    # Define the model
    model = UNet3DTimestapHidden(
        in_channels=6,
        out_channels=6,
        skip_connections=False,
        layers=3,
        num_steps=1000,
        cross_dim=64,
    )

    # Create a random input tensor with shape (batch_size, channels, depth, height, width)
    input_tensor = torch.randn(2, 6, 51, 17, 2)
    timesteps = torch.randint(0, 1000, (2,))
    hidden_state = torch.randn(2, 64, 16, 16, 16)

    # Print the shape of the input tensor
    print("Input shape:", input_tensor.shape)

    # Pass the input tensor through the model
    output_tensor = model(input_tensor, timesteps, hidden_state)

    # Print the shape of the output tensor
    print("Output shape:", output_tensor.shape)

    # Test the encode function
    encoded_tensor = model.encode(input_tensor, timesteps)

    # Print the shape of the encoded tensor
    print("Encoded shape:", encoded_tensor.shape)


if __name__ == "__main__":
    main()
