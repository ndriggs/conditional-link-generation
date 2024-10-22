from torch.utils.data import DataLoader
import torch
import argparse
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # options: mlp, cnn, transformer_encoder, reformer, gnn
    parser.add_argument('--model', type=str, default='cnn')
    # only applicable to mlp
    parser.add_argument('--hidden_size', type=int, default=100)
    # only applicable to mlp
    parser.add_argument('--dropout', type=float, default=0.3)
    # options: clip_then_normalize, abs_log_then_scale (only applicable to cnn and mlp)
    # other option: remove_cancelations (only applicable to transformer_encoder, reformer, and gnn)
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    # options: 2 or 3 (only applicable to cnn)
    parser.add_argument('--kernel_size', type=int, default=2)
    # only applicable to cnn
    parser.add_argument('--layer_norm', type=bool, default=True)
    parser.add_argument('--regression', type=bool, default=True)
    return parser.parse_args()

def main():
    args = parse_args()

    train_data = np.load('train.npy')
    val_data = np.load('val.npy')
    test_data = np.load('test.npy')

    if args.preprocessing == 'clip_then_normalize' :
        pass 
    elif args.preprocessing == ''