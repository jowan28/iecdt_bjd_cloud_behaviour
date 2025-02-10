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


# TODO: VALIDATION, CONFIG FILE 

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

    train_data_loader, val_data_loader = iecdt_lab.latent_regressor_loader(train=True,
                                                                           tiles_path=cfg.tiles_path, 
                                                                           train_metadata=cfg.train_metadata,
                                                                           val_metadata=cfg.val_metadata,
                                                                           batch_size=cfg.batch_size,
                                                                           data_transforms=data_transforms,
                                                                           dataloader_workers=cfg.dataloade_workers,)

    
    model = iecdt_lab.latent_regressor(cfg.latent_dim)
    model = model.to(cfg.device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(cfg.epochs):
        model.train()
        for i, (batch, labels) in enumerate(train_data_loader):
            optimizer.zero_grad()

            batch = batch.to(cfg.device)
            preds = model(batch)
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
        
        eval_fig, val_mse = validation(cfg, model, val_data_loader, data_stats)
        wandb.log({"predictions": eval_fig, "loss/val": val_mse})

        if cfg.smoke_test:
            break

    


if __name__ == "__main__":
    main()


