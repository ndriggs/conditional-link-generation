import sklearn
from link_generation.models.curiousity_models import GNN
from link_generation.potholders.utils import *
import torch.nn as nn
import torch
import lightning as pl
from torch.optim.lr_scheduler import ExponentialLR

class VAE(pl.LightningModule) :
    def __init__(self, mean, scale, 
                 num_gnn_layers=5, hidden_channels=32, num_heads=8, 
                 latent_embedding_size=2, mlp_hidden_size=400, potholder_size=9,
                 k=2, device='cuda') :
        super(VAE, self).__init__()

        # when training on braids
        # self.encoder = GNN(hidden_channels=hidden_channels, num_heads=num_heads, 
        #                    num_layers=num_gnn_layers, dropout=0,
        #                    classification=False, both=False, ohe_inverses=True, 
        #                    double_features=True, laplacian=False, k=1, 
        #                    return_features=True, braid_index=7)
        self.encoder = GNN(hidden_channels=hidden_channels, num_heads=num_heads, 
                           num_layers=num_gnn_layers, double_features=True, k=k, 
                           return_features=True, potholder_size=potholder_size)
        self.mu_transform = nn.Linear(hidden_channels*(2**(num_gnn_layers-1))*num_heads, latent_embedding_size)
        self.logvar_transform = nn.Linear(hidden_channels*(2**(num_gnn_layers-1))*num_heads, latent_embedding_size)

        self.decoder = nn.Sequential(
            nn.Linear(latent_embedding_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, potholder_size**2 - 2),
            nn.Sigmoid()
        )

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.mean = torch.tensor(mean).to(device)
        self.scale = torch.tensor(scale).to(device)

    def forward(self, x) :
        # encode 
        x = self.encoder(x)
        mu = self.mu_transform(x)
        logvar = self.logvar_transform(x)

        # reparameterize
        z = self.reparameterize(mu, logvar)

        # decode 
        x_hat = self.decoder(z)

        # round to nearest knot using the straight-through gradient estimator
        # since round has zero gradient everywhere
        knot = x_hat + (x_hat.round() - x_hat).detach()

        # make sure they all got correctly rounded
        assert torch.all(((knot == torch.ones_like(knot)) + (knot == torch.zeros_like(knot))))

        # calculate Goeritz matrix and invariants
        P = state_to_potholder_pytorch(knot)
        G = potholder_to_goeritz_pytorch(P)
        invariants = goeritz_to_invariants(G)

        return z, mu, logvar, invariants

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def latent_to_invariants(self, z) :
        # decode 
        x_hat = self.decoder(z)

        # round to nearest knot using the straight-through gradient estimator
        # since round has zero gradient everywhere
        knot = x_hat + (x_hat.round() - x_hat).detach()

        # make sure they all got correctly rounded
        assert torch.all(((knot == torch.ones_like(knot)) + (knot == torch.zeros_like(knot))))

        # calculate Goeritz matrix and invariants
        P = state_to_potholder_pytorch(knot)
        G = potholder_to_goeritz_pytorch(P)
        invariants = goeritz_to_invariants(G)

        return invariants
    
    def compute_mse_loss(self, invariants, batch) :
        sig, log_det = invariants
        y_hat = torch.cat([sig.unsqueeze(1), log_det.unsqueeze(1)], dim=1)
        y_hat_scaled = (y_hat - self.mean) / self.scale
        return self.mse_loss(y_hat_scaled, batch.y)
    
    def compute_invariant_l1_loss(self, invariants, batch) :
        sig, log_det = invariants
        unscaled_y = batch.y*self.scale + self.mean
        return self.l1_loss(sig, unscaled_y[:,0]), self.l1_loss(log_det, unscaled_y[:,1])

    def training_step(self, batch, batch_idx):
        z, mu, logvar, invariants = self(batch)
        
        # compute the MSE loss
        mse_loss = self.compute_mse_loss(invariants, batch)

        # compute the KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # combine the losses
        loss = mse_loss + kld_loss

        # compute the unscaled l1 losses
        sig_loss, log_det_loss = self.compute_invariant_l1_loss(invariants, batch)

        # log and return
        self.log('train_sig_loss', sig_loss)
        self.log('train_logdet_loss', log_det_loss)
        self.log('train_mse_loss', mse_loss)
        self.log('train_kld_loss', kld_loss)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        z, mu, logvar, invariants = self(batch)
        
        # compute the MSE loss
        mse_loss = self.compute_mse_loss(invariants, batch)

        # compute the KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # combine the losses
        loss = mse_loss + kld_loss

        # compute the unscaled l1 losses
        sig_loss, log_det_loss = self.compute_invariant_l1_loss(invariants, batch)

        # log and return
        self.log('val_sig_loss', sig_loss)
        self.log('val_logdet_loss', log_det_loss)
        self.log('val_mse_loss', mse_loss)
        self.log('val_kld_loss', kld_loss)
        self.log('val_loss', loss)
        
        return loss

    def test_step(self, batch, batch_idx):
        z, mu, logvar, invariants = self(batch)
        
        # compute the MSE loss
        mse_loss = self.compute_mse_loss(invariants, batch)

        # compute the KL divergence loss
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # combine the losses
        loss = mse_loss + kld_loss

        # log and return
        self.log('test_mse_loss', mse_loss)
        self.log('test_kld_loss', kld_loss)
        self.log('test_loss', loss)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "exp_lr"
            }
        }