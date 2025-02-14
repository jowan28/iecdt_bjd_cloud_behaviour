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
from torch.nn import functional as F

import iecdt_lab
import iecdt_lab.VAE


def plot_reconstructions(batch, reconstructions, data_stats, max_images=8, normalise=True):
    fig, axes = plt.subplots(2, max_images, figsize=(15, 5))
    batch, reconstructions = batch[:max_images], reconstructions[:max_images]
    for i, (img, recon) in enumerate(zip(batch, reconstructions)):
        img = img.permute(1, 2, 0).cpu().numpy()
        recon = recon.permute(1, 2, 0).cpu().numpy()
        if normalise:
            img = img * data_stats["rgb_std"] + data_stats["rgb_mean"]
            recon = recon * data_stats["rgb_std"] + data_stats["rgb_mean"]
        axes[0, i].imshow(img)
        axes[0, i].set_title("Original")
        axes[0, i].axis("off")
        axes[1, i].imshow(recon)
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis("off")

    fig.tight_layout()
    return fig


def validation(cfg, model, test_data_loader, data_stats, normalise=True):
    model.eval()
    running_mse = 0
    num_batches = len(test_data_loader)
    with torch.no_grad():
        for i, (batch, _) in enumerate(test_data_loader):
            batch = batch.to(cfg.device)
            loss, reconstructions = model.batch_loss(batch, return_reconstructions=True, beta=cfg.vae_beta)
            running_mse += loss.cpu().numpy()
            

            if i == 0:
                fig = plot_reconstructions(batch, reconstructions, data_stats, normalise=normalise)

            if cfg.smoke_test and i == 10:
                num_batches = i + 1
                break

    val_mse = running_mse / num_batches
    return fig, val_mse


def wandb_initialiser(cfg: DictConfig, id_in: str):
    wandb.init(
        id=id_in,
        resume="allow",
        project=cfg.wandb.project,
        group=cfg.name,
        config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True),
        mode=cfg.wandb.mode,
    )


@hydra.main(version_base=None, config_path="config_ae", config_name="train")
def main(cfg: DictConfig):
    print("Yah lets go")

    resume = True
    # Set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    wandb.login(key=os.environ["WANDB_API_KEY"])
    if resume:
        # Generate ID to store and resume run.
        if os.path.exists("wandb_id.txt"):
            with open("wandb_id.txt", "r") as f:
                wandb_id = f.read().strip()
        else:
            wandb_id = wandb.util.generate_id()
    else:
        wandb_id = wandb.util.generate_id()
    wandb_initialiser(cfg, wandb_id)

    data_stats = np.load(cfg.train_rgb_stats)
    if cfg.normalise:
        data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=data_stats["rgb_mean"], std=data_stats["rgb_std"]
                ),
            ]
        )
    else:
        data_transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
            ]
        )

    train_data_loader, val_data_loader = iecdt_lab.data_loader.get_data_loaders(
        tiles_path=cfg.tiles_path,
        train_metadata=cfg.train_metadata,
        val_metadata=cfg.val_metadata,
        batch_size=cfg.batch_size,
        data_transforms=data_transforms,
        dataloader_workers=cfg.dataloader_workers,
    )

    model = iecdt_lab.VAE.CNNVariationalAutoencoder(latent_dim=cfg.latent_dim, lim_decoder=cfg.lim_decoder)
    if resume and os.path.exists("model.pth"):
        print("loading previous parameters")
        model.load_state_dict(torch.load("model.pth"))
    model = model.to(cfg.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    best_val_mse = float("inf")
    print("loaded and about to go!")
    for epoch in range(cfg.epochs):
        print("epoch")
        model.train()
        for i, (batch, _) in enumerate(train_data_loader):
            print("batch")
            optimizer.zero_grad()

            batch = batch.to(cfg.device)
            loss = model.batch_loss(batch, beta=cfg.vae_beta)
            loss.backward()
            optimizer.step()

            if i % cfg.log_freq == 0:
                logging.info(
                    f"Epoch {epoch}/{cfg.epochs} Batch {i}/{len(train_data_loader)}: Loss={loss.item():.2f}"
                )
                wandb.log({"loss/train": loss.item()})

            if cfg.smoke_test and i == 50:
                break

        eval_fig, val_mse = validation(cfg, model, val_data_loader, data_stats, normalise=cfg.normalise)
        wandb.log({"predictions": eval_fig, "loss/val": val_mse})

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            torch.save(model.state_dict(), "best_model.pth")
            logging.info(f"Saved best model with val_mse: {val_mse:.2f}")

        if cfg.smoke_test:
            break

        with open("wandb_id.txt", "w") as f:
            f.write(wandb_id)
        torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()

"""
I need to test with normalisation on limit off
normalisation off limit on
"""