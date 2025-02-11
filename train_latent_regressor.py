import iecdt_lab
import logging
import os
import random

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

import iecdt_lab.latent_regressor


# TODO: CONFIG FILE


def lr_validation(cfg, model, test_data_loader, data_stats):
    model.eval()
    running_mse = 0
    num_batches = len(test_data_loader)
    with torch.no_grad():
        for i, (batch, labels) in enumerate(test_data_loader):
            batch = batch.to(cfg.device)
            predictions = model(batch)
            running_mse += torch.mean((predictions - labels) ** 2).cpu().numpy()

            if cfg.smoke_test and i == 10:
                num_batches = i + 1
                break
    val_mse = running_mse / num_batches
    return val_mse


@hydra.main(version_base=None, config_path="config_lr", config_name="train")
def main(cfg: DictConfig):
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    wandb.login(key=os.environ["WANDB_API_KEY"])

    # Generate ID
    wandb_id = wandb.util.generate_id()

    wandb.init(
        id=wandb_id,
        resume="allow",
        project=cfg.wandb.project,
        group=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
    )

    data_stats = np.load(cfg.train_rgb_stats)
    data_transforms = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=data_stats["rgb_mean"], std=data_stats["rgb_std"]
            ),
        ]
    )

    train_data_loader, val_data_loader = (
        iecdt_lab.latent_regressor_data_loader.get_latent_regressor_data_loader(
            cfg, transform=data_transforms
        )
    )

    model = iecdt_lab.latent_regressor.Latent_regressor(cfg.latent_dim)
    model = model.to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        for i, (batch, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()
            logging.info(f"batch shape: {batch.shape}")
    
            batch = batch.to(cfg.device)
            preds = model(batch.T)
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            if i % cfg.log_freq == 0:
                logging.info(
                    f"Epoch {epoch}/{cfg.epochs} Batch {i}/{len(train_data_loader)}: Loss={loss.item():.2f}"
                )
                wandb.log({"loss/train": loss.item()})

            if cfg.smoke_test and i == 50:
                break

        torch.save(model.state_dict(), "model.pth")

        val_mse = lr_validation(cfg, model, val_data_loader, data_stats)
        wandb.log({"loss/val": val_mse})

        if cfg.smoke_test:
            break


if __name__ == "__main__":
    main()
