from torch import nn
from torch.nn import functional as F
import torch

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc_mu = nn.Linear(128 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(128 * 8 * 8, latent_dim)
        self.decoder_fc = nn.Linear(latent_dim, 128 * 8 * 8)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.decoder_fc(z)
        h = h.view(-1, 128, 8, 8)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    def batch_loss(self, batch, return_reconstructions=False):
        reconstructions, mu, logvar = self.forward(batch)
        recon_loss = F.mse_loss(reconstructions, batch, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if return_reconstructions:
            return recon_loss + kl_loss, reconstructions
        return recon_loss + kl_loss
    

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
        self.fc_mu = nn.Linear(128 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(128 * 32 * 32, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        h = x.flatten(start_dim=1)  # Flatten all dimensions except batch
        mu = self.fc_mu(h)
        log_var = self.fc_logvar(h)
        return mu, log_var
    


class Decoder(nn.Module):
    def __init__(self, input_channels: int, latent_dim: int, lim_decoder: bool):
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
        self.output_activation = nn.Tanh()
        self.lim_decoder = lim_decoder

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = self.linear(z)
        z = z.view(-1, 128, 32, 32)  # Reshape to (batch_size, 128, 32, 32)
        z = F.relu(self.conv1(z))
        z = F.relu(self.conv2(z))
        z = F.relu(self.conv3(z))
        z = self.conv4(z)
        if self.lim_decoder:
            z = 0.5 * self.output_activation(z) + 0.5
        return z


class CNNVariationalAutoencoder(nn.Module):
    def __init__(self, latent_dim: int, input_channels: int = 3, lim_decoder: bool = False) -> None:
        super().__init__()
        self.encoder = Encoder(input_channels, latent_dim)
        self.decoder = Decoder(input_channels, latent_dim, lim_decoder)

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

    def batch_loss(self, batch: torch.Tensor, beta: float = 1.0,  return_reconstructions: bool = False) -> torch.Tensor:
        reconstructions, mu, logvar = self.forward(batch)
        recon_loss = F.mse_loss(reconstructions, batch, reduction='sum')
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        if return_reconstructions:
            return recon_loss + beta * kl_loss, reconstructions
        return recon_loss + kl_loss