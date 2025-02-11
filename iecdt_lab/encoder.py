import torch
from iecdt_lab.autoencoder import CNNAutoencoder

"""code to read in the trained ae from a file, remove the decoder.
this function will then recieve a tile and output the embedding"""


def load_encoder(model_path, latent_dim):
    autoencoder = CNNAutoencoder()
    # autoencoder.load_state_dict(torch.load(model_path))
    encoder = autoencoder.encoder
    encoder.eval()
    return encoder


# model_path = "model.pth"
