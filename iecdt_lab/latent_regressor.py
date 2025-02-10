"""
This file will contain functionality to aim to identify a the set of basis vectors of an embedding space that represent the cloud cover metric.

This is to be done via training a model on the embedding space as input, and cloud cover as the label. The network will consist of a single fully connected layer of one node, by analysising the trained weights we can determine the key embedding state vectors that make up the cloud coverage component.
"""

import torch 
from torch import nn
# Class based on nn.module with defined layers 

class Latent_regressor(nn.Module):
    def __init__(self, latent_dim):
        """
        Initilising latent regressor class. This consists of a single layer with a single node 
        """
        super().__init__()
        self.model = nn.sequential(
            nn.Linear(latent_dim, 1, bias=False),
        )

    def forward(self, x):
        return self.model(x)






# Function to plot 
