import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

import iecdt_lab


class GOESRGBTileLatents(Dataset):
    def __init__(self, cfg, transform, train=False, val=False, test=False) -> None:
        """
        Arguments:
        cfg -- config file
        transform -- transform to be applied to large images to get tiles
        train -- bool to indicate if train dataset is required
        val -- bool to indicate if val dataset is required
        test -- bool to indicate if test dataset is required
        """

        if sum([train, val, test]) > 1:
            raise ValueError(
                "Only one of 'train', 'val', or 'test' can be True at a time."
            )

        metadata = {
            train: cfg.train_metadata,
            val: cfg.val_metadata,
            test: cfg.test_metadata,
        }.get(True)

        self.tiles_dataset = iecdt_lab.data_loader.GOESRGBTiles(
            tiles_file=cfg.tiles_path,
            metadata_file=metadata,
            cloud_fraction_threashold=cfg.cloud_fraction_threashold,
            transform=transform,
        )

        # Get trained encoder
        self.encoder = iecdt_lab.encoder.load_encoder(cfg.encoder_model_path)

        # Does tile need to_device() ??
        # self.device = cfg.device()

    def __len__(self) -> int:
        return len(self.tiles_dataset)

    def __getitem__(self, index: int):
        self.encoder.eval()

        # Get target tile
        tile, metadata = self.tiles_dataset[index]

        # Get latent representation
        with torch.no_grad():
            latent = self.encoder(tile)

        # Get cloud fraction
        cloud_fraction = metadata.cloud_fraction

        return latent, cloud_fraction


def get_latent_regressor_data_loader(cfg, transform):
    """
    Returns train and val data loaders
    """
    train_ds = GOESRGBTileLatents(cfg, transform, train=True)

    train_data_loader = torch.utils.data.DataLoader(
        train_ds, cfg.batch_size, shuffle=True, num_workers=cfg.dataloader_workers
    )

    val_ds = GOESRGBTileLatents(cfg, transform, val=True)

    val_data_loader = torch.utils.data.DataLoader(
        val_ds, cfg.batch_size, shuffle=True, num_workers=cfg.dataloader_workers
    )

    return train_data_loader, val_data_loader
