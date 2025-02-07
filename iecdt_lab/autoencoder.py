from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        batchnorm: bool = False,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=3,
            stride=2,
            padding=1,
        )
        self.batchnorm = nn.Identity()
        if batchnorm:
            self.batchnorm = nn.BatchNorm2d(out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.gelu(self.conv(x))
        x = self.batchnorm(x)
        return x


class Encoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        # Input shape: (batch_size, 1, 256, 256)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch_size, 32, 128, 128)
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch_size, 64, 64, 64)
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
        )  # (batch_size, 128, 32, 32)
        self.linear = nn.Linear(32 * 32 * 128, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.flatten(start_dim=1)  # Flatten all dimensions except batch
        x = self.linear(x)
        return x


class Decoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int):
        super().__init__()
        self.linear = nn.Linear(latent_dim, 32 * 32 * 128)
        self.conv1 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=128,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv2 = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv3 = nn.ConvTranspose2d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
        )
        self.conv4 = nn.ConvTranspose2d(
            in_channels=32,
            out_channels=input_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.linear(z)
        z = z.view(-1, 128, 32, 32)  # Reshape to (batch_size, 128, 32, 32)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = self.conv4(z)
        return z


class CNNAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, input_channels: int = 3) -> None:
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        x_reconstructed = self.decoder(z)
        return x_reconstructed
