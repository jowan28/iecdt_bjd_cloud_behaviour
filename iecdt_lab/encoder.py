import torch
import os
from iecdt_lab.autoencoder import CNNAutoencoder
from omegaconf import DictConfig, OmegaConf
import hydra
import torchvision


"""code to read in the trained ae from a file, remove the decoder.
this function will then recieve a tile and output the embedding"""


@hydra.main(version_base=None, config_path="config_ae", config_name="train")
def remove_decoder(cfg: DictConfig):
    """model_path is the path to the trained autoencoder model
    returns the encoder model class."""
    model = CNNAutoencoder(latent_dim=cfg.latent_dim)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(cfg.device)

    encoder = model.encoder
    encoder.eval()
    torch.save(
        encoder.state_dict(), "/home/users/dash/iecdt_bjd_cloud_behaviour/encoder.pth"
    )
    return


@hydra.main(version_base=None, config_path="config_ae", config_name="train")
def load_encoder(cfg: DictConfig):
    """model_path is the path to the trained autoencoder model
    returns the encoder model class."""
    model = CNNAutoencoder(latent_dim=cfg.latent_dim)
    encoder = model.encoder
    encoder.load_state_dict(torch.load(encoder_path))
    encoder.eval()
    return encoder


# model.load_state_dict(torch.load("path/to/experiment/model.pth"))

path = "/home/users/dash/iecdt_bjd_cloud_behaviour/"
model_path = os.path.join(path, "model.pth")
encoder_path = os.path.join(path, "encoder.pth")
remove_decoder()
load_encoder()
