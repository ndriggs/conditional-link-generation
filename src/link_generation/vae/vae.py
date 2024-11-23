from link_generation.models.curiousity_models import GNN
from link_generation.potholders.utils import *
import torch.nn as nn
import torch
import lightning as pl

class VAE(pl.LightningModule) :
    def __init__(self, num_gnn_layers=5, latent_embedding_size=2, mlp_hidden_size=500, potholder_size=9) :

        self.encoder = GNN(num_layers=num_gnn_layers)
        self.mu_transform = nn.Linear(2**(num_gnn_layers-1), latent_embedding_size)
        self.logvar_transform = nn.Linear(2**(num_gnn_layers-1), latent_embedding_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_embedding_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, potholder_size**2),
            nn.Sigmoid()
        )
