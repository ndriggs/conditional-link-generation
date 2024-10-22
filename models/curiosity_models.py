import torch
import torch.nn as nn
import lightning as pl
from torch.optim.lr_scheduler import ExponentialLR
import math

class MLP(pl.LightningModule):
    def __init__(self, lk_matrix_size, hidden_size, 
                 dropout, num_invariants=1):
        super(MLP, self).__init__()

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.fc1 = nn.Linear(lk_matrix_size**2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_invariants)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x) :
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse_loss(y_hat, y)
        l1_loss = self.l1_loss(y_hat, y)
        self.log('val_mse_loss', mse_loss)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse_loss(y_hat, y)
        l1_loss = self.l1_loss(y_hat, y)
        self.log('test_mse_loss', mse_loss)
        self.log('test_l1_loss', l1_loss)

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


class CNN(pl.LightningModule):
    def __init__(self, lk_matrix_size: int, kernel_size: int, 
                 layer_norm: bool, num_invariants: int = 1) :
        super(CNN, self).__init__()

        self.layer_norm = layer_norm

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

        self.relu = nn.ReLU()

        padding = int(kernel_size == 3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                              kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32,
                              kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64,
                              kernel_size=kernel_size, stride=1,
                              padding=padding)

        if layer_norm :
          self.norm1 = nn.LayerNorm([16, lk_matrix_size-(3-kernel_size),
                                    lk_matrix_size-(3-kernel_size)])
          self.norm2 = nn.LayerNorm([32, lk_matrix_size-2*(3-kernel_size),
                                    lk_matrix_size-2*(3-kernel_size)])
          self.norm3 = nn.LayerNorm([64, lk_matrix_size-3*(3-kernel_size),
                                    lk_matrix_size-3*(3-kernel_size)])

        self.fc1 = nn.Linear(64*(lk_matrix_size-3*(3-kernel_size))**2, 1000)
        self.fc2 = nn.Linear(1000, num_invariants)

    def forward(self, x) :
        # first convolution layer
        x = self.conv1(x)
        if self.layer_norm : 
          x = self.norm1(x)
        x = self.relu(x)

        # second convolutional layer 
        x = self.conv2(x)
        if self.layer_norm :
          x = self.norm2(x)
        x = self.relu(x)

        # third convolutional layer 
        x = self.conv3(x)
        if self.layer_norm : 
          x = self.norm3(x) 
        x = self.relu(x)
        
        # feed forward layers
        x = x.view(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse_loss(y_hat, y)
        l1_loss = self.l1_loss(y_hat, y)
        self.log('val_mse_loss', mse_loss)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        mse_loss = self.mse_loss(y_hat, y)
        l1_loss = self.l1_loss(y_hat, y)
        self.log('test_mse_loss', mse_loss)
        self.log('test_l1_loss', l1_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)
        scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": f"{self.lr_schedule}_lr"
            }
        }


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, dim_feedforward, max_seq_length):
        super(TokenPredictor, self).__init__()
        
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.final_layer = nn.Linear(d_model, 1)

    def forward(self, src):
        # src shape: (seq_len, batch_size)
        
        # Embed the input tokens
        src = self.embed(src) * math.sqrt(self.d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Pass through the transformer encoder
        output = self.transformer_encoder(src)
        
        # Take the mean of the sequence dimension
        output = output.mean(dim=0)
        
        # Pass through the final linear layer
        output = self.final_layer(output)
        
        # Squeeze to get a single value
        return output.squeeze(-1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]
