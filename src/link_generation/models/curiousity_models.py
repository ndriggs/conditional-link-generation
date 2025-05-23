import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as pl
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from reformer_pytorch import ReformerLM
from torch_geometric.nn.conv import TransformerConv
from torch_geometric.nn.pool import global_mean_pool, global_max_pool
from .utils import topk_accuracy
import numpy as np
import math

BRAID_INDEX = 7

class NaiveModel(pl.LightningModule):
    '''
    Naively learns a linear model for computing signature based on the 
    difference between number of sigmas and sigmas inverses.

    Since adding the same sigma over and over again causes the signature 
    to go down and adding the same inverse repeatedly causes the signature 
    to go up, we'd expect the weight to be negative and bias close to zero. 
    '''
    def __init__(self):
        super(NaiveModel, self).__init__()
        self.linear = nn.Linear(1,1, bias=True)
        self.l1_loss = nn.L1Loss()

    def forward(self, x) :
        x, length = x
        x = torch.sum(torch.sign(x), dim=1).unsqueeze(1).to(torch.float32)
        return self.linear(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('test_l1_loss', l1_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.01)
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

    

class MLP(pl.LightningModule):
    def __init__(self, hidden_size, dropout, lk_matrix_size=21, 
                 num_invariants=1, classification=False, num_classes=75):
        super(MLP, self).__init__()

        self.classification = classification
        self.num_classes = num_classes

        self.l1_loss = nn.L1Loss()
        if classification :
            self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.fc1 = nn.Linear(lk_matrix_size**2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        if classification :
            self.fc3 = nn.Linear(hidden_size, num_classes)
        else :
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
        if self.classification :
            y = (y + ((self.num_classes - 1) / 2)).to(torch.int64) # shift the signatures so they start at 0
            loss = self.cross_entropy(y_hat.squeeze(1), y)
        else : # regression
            loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            # convert signatures to class ids 
            target_classes = (y + ((self.num_classes - 1) / 2)).to(torch.int64)
            top1_acc, top5_acc = topk_accuracy(y_hat, target_classes)
            self.log('test_top1_acc', top1_acc.item())
            self.log('test_top5_acc', top5_acc.item())
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
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
    def __init__(self, kernel_size: int, layer_norm: bool, 
                 lk_matrix_size: int = 21, num_invariants: int = 1,
                 init_hidden_size=2, classification=False, num_classes=75) :
        super(CNN, self).__init__()

        self.layer_norm = layer_norm
        self.classification = classification
        self.num_classes = num_classes

        self.l1_loss = nn.L1Loss()
        if classification :
            self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.relu = nn.ReLU()

        padding = int(kernel_size == 3)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=init_hidden_size,
                              kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.conv2 = nn.Conv2d(in_channels=init_hidden_size, out_channels=init_hidden_size*2,
                              kernel_size=kernel_size, stride=1,
                              padding=padding)
        self.conv3 = nn.Conv2d(in_channels=init_hidden_size*2, out_channels=init_hidden_size*3,
                              kernel_size=kernel_size, stride=1,
                              padding=padding)

        if layer_norm :
          self.norm1 = nn.LayerNorm([init_hidden_size, lk_matrix_size-(3-kernel_size),
                                    lk_matrix_size-(3-kernel_size)])
          self.norm2 = nn.LayerNorm([init_hidden_size*2, lk_matrix_size-2*(3-kernel_size),
                                    lk_matrix_size-2*(3-kernel_size)])
          self.norm3 = nn.LayerNorm([init_hidden_size*3, lk_matrix_size-3*(3-kernel_size),
                                    lk_matrix_size-3*(3-kernel_size)])

        self.fc1 = nn.Linear(init_hidden_size*3*(lk_matrix_size-3*(3-kernel_size))**2, 1000)
        if self.classification :
            self.fc2 = nn.Linear(1000, num_classes)
        else : 
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
        if self.classification :
            y = (y + ((self.num_classes - 1) / 2)).to(torch.int64) # shift the signatures so they start at 0
            loss = self.cross_entropy(y_hat.squeeze(1), y)
        else : # regression
            loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            # convert signatures to class ids 
            target_classes = (y + ((self.num_classes - 1) / 2)).to(torch.int64)
            top1_acc, top5_acc = topk_accuracy(y_hat, target_classes)
            self.log('test_top1_acc', top1_acc.item())
            self.log('test_top5_acc', top5_acc.item())
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
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


class TransformerEncoder(pl.LightningModule):
    def __init__(self, vocab_size, d_model, nhead, num_encoder_layers, 
                 dim_feedforward, max_seq_length, classification=False,
                 num_classes=75, warmup_steps=4000):
        super(TransformerEncoder, self).__init__()

        self.classification = classification
        self.num_classes = num_classes
        self.warmup_steps = warmup_steps

        self.l1_loss = nn.L1Loss()
        if classification :
            self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        if self.classification :
            self.final_layer = nn.Linear(d_model, num_classes)
        else : # regression
            self.final_layer = nn.Linear(d_model, 1)

    def forward(self, src_and_lengths):
        src, src_lengths = src_and_lengths
        # src shape: (batch_size, seq_len)
        
        # Embed the input tokens
        src = self.embed(src) * math.sqrt(self.d_model)
        # now (batch_size, seq_len, d_model)
        
        # Add positional encoding
        src = self.pos_encoder(src)
        
        # Transpose to (seq_len, batch_size, d_model) for transformer
        src = src.transpose(0, 1)
        
        # Create mask based on src_lengths
        src_key_padding_mask = (torch.arange(src.size(0)).unsqueeze(0).to('cuda:0') >= src_lengths.unsqueeze(1)).to(src.device)
        
        # Pass through the transformer encoder
        output = self.transformer_encoder(src, src_key_padding_mask=src_key_padding_mask)
        
        # Take the mean of the sequence dimension
        output = output.mean(dim=0)
        
        # Pass through the final linear layer
        output = self.final_layer(output)
        
        # Squeeze to get a single value
        return output.squeeze(-1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            y = (y + ((self.num_classes - 1) / 2)).to(torch.int64) # shift the signatures so they start at 0
            loss = self.cross_entropy(y_hat.squeeze(1), y)
        else : # regression
            loss = self.l1_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2))
        l1_loss = self.l1_loss(y_hat, y)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            # convert signatures to class ids 
            target_classes = (y + ((self.num_classes - 1) / 2)).to(torch.int64)
            top1_acc, top5_acc = topk_accuracy(y_hat, target_classes)
            self.log('test_top1_acc', top1_acc.item())
            self.log('test_top5_acc', top5_acc.item())
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2))
        l1_loss = self.l1_loss(y_hat, y)
        self.log('test_l1_loss', l1_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)

        def noam_lambda(step):
            return (self.d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * self.warmup_steps ** -1.5)

        scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "noam_lr"
            }
        }

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # repeat the positional encoding (originally (max_len, d_model)) along 
        # the batch dimension
        return x + self.pe[:x.size(1), :].unsqueeze(0).repeat(x.size(0), 1, 1)

class Reformer(pl.LightningModule) :
    def __init__(self, vocab_size, d_model, nhead, num_layers,
                 max_seq_len, classification=False,
                 num_classes=75, warmup_steps=4000):
        super(Reformer, self).__init__()

        self.classification = classification
        self.num_classes = num_classes
        self.d_model = d_model
        self.warmup_steps = warmup_steps

        self.l1_loss = nn.L1Loss()
        if classification :
            self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)

        self.reformer = ReformerLM(vocab_size, d_model, num_layers, max_seq_len=max_seq_len, 
                                   heads = nhead, causal = False, use_full_attn = True,  
                                   return_embeddings = True, axial_position_emb = True)

        if self.classification : 
            self.fc = nn.Linear(d_model, num_classes)
        else : 
            self.fc = nn.Linear(d_model, 1)

    def forward(self, x_and_lengths) :
        x, lengths = x_and_lengths # originally a tuple is passed in 
        x = self.reformer(x)
        x = self.fc(x[torch.arange(x.shape[0]),lengths-1,:])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            y = (y + ((self.num_classes - 1) / 2)).to(torch.int64) # shift the signatures so they start at 0
            loss = self.cross_entropy(y_hat.squeeze(1), y)
        else : # regression
            loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('val_l1_loss', l1_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        if self.classification :
            assert len(y_hat.shape) == 2
            # convert signatures to class ids 
            target_classes = (y + ((self.num_classes - 1) / 2)).to(torch.int64)
            top1_acc, top5_acc = topk_accuracy(y_hat, target_classes)
            self.log('test_top1_acc', top1_acc.item())
            self.log('test_top5_acc', top5_acc.item())
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), y)
        self.log('test_l1_loss', l1_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=.001)

        def noam_lambda(step):
            return (self.d_model ** -0.5) * min((step + 1) ** -0.5, (step + 1) * self.warmup_steps ** -1.5)

        scheduler = LambdaLR(optimizer, lr_lambda=noam_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
                "name": "noam_lr"
            }
        }


class GNN(pl.LightningModule):
    def __init__(self, hidden_channels=16, num_heads=4, num_layers=5, dropout=0,
                 classification=False, both=True, pos_neg=False, ohe_inverses=True, 
                 laplacian=False, k=0, double_features=True, return_features=False, 
                 num_classes=75, braid_index=7, potholder_size=0):
        super(GNN, self).__init__()
        self.return_features = return_features

        if potholder_size > 0 :
            num_node_features = 2*potholder_size + 2 + k
        elif laplacian :
            num_node_features = k
        elif both and ohe_inverses :
            num_node_features = (braid_index-1)*2 + 2 + k
        elif both :
            num_node_features = braid_index + 1 + k
        elif pos_neg :
            num_node_features = 2 + k
        elif ohe_inverses :
            num_node_features = (braid_index-1)*2 + k
        else :
            num_node_features = braid_index-1 + k

        self.conv_layers = nn.ModuleList([
            TransformerConv(num_node_features, hidden_channels, heads=num_heads)
        ])
        for layer_num in range(2, num_layers+1) :
            if double_features : # double the number of features at each layer
                self.conv_layers.append(TransformerConv((2**(layer_num-2))*hidden_channels*num_heads, 
                                                        (2**(layer_num-1))*hidden_channels, heads=num_heads))
            else : # constant number of features at each layer
                self.conv_layers.append(TransformerConv(hidden_channels*num_heads, 
                                                        hidden_channels, heads=num_heads))
        
        # calculate the input dimension for the linear layer
        if double_features :
            fc_in_dim = (2**(num_layers-1))*hidden_channels*num_heads
        else : 
            fc_in_dim = hidden_channels*num_heads

        if classification :
            self.fc = torch.nn.Linear(fc_in_dim, num_classes)
        else :
            self.fc = torch.nn.Linear(fc_in_dim, 1)

        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self.classification = classification
        self.l1_loss = nn.L1Loss()
        if classification :
            self.cross_entropy = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.num_classes = num_classes

    def forward(self, data):
        # first conv layer 
        x = self.conv_layers[0](data.x, data.edge_index)
        x = F.relu(x)
        
        for conv_layer in self.conv_layers[1:] :
            x = self.dropout(x)
            x = conv_layer(x, data.edge_index)
            x = F.relu(x)

        # pooling and linear layer
        x = global_max_pool(x, data.batch) # TODO: experiment with concating max and mean
        if self.return_features :
            return x
        else :
            return self.fc(x)

    def training_step(self, batch, batch_idx):
        y_hat = self(batch)
        if self.classification :
            y = (batch.y + ((self.num_classes - 1) / 2)).to(torch.int64) # shift the signatures so they start at 0
            loss = self.cross_entropy(y_hat.squeeze(1), y)
        else : # regression
            loss = self.l1_loss(y_hat.squeeze(1), batch.y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        y_hat = self(batch)
        if self.classification :
            assert len(y_hat.shape) == 2
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), batch.y)
        # add the batch_size= to suppress warnings, even though that's not the actual batch size
        self.log('val_l1_loss', l1_loss, batch_size=batch.x.shape[0])

    def test_step(self, batch, batch_idx):
        y_hat = self(batch)
        if self.classification :
            assert len(y_hat.shape) == 2
            # convert signatures to class ids 
            target_classes = (batch.y + ((self.num_classes - 1) / 2)).to(torch.int64)
            top1_acc, top5_acc = topk_accuracy(y_hat, target_classes)
            self.log('test_top1_acc', top1_acc.item(), batch_size=batch.x.shape[0])
            self.log('test_top5_acc', top5_acc.item(), batch_size=batch.x.shape[0])
            pred_classes = y_hat.argmax(1) # preserve the batch dim, armax over classes
            # convert top predicted class labels to signatures, so that we can compute 
            # regression loss 
            y_hat = (pred_classes - ((self.num_classes - 1)/2)).unsqueeze(1)
        l1_loss = self.l1_loss(y_hat.squeeze(1), batch.y) 
        self.log('test_l1_loss', l1_loss, batch_size=batch.x.shape[0])

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
        