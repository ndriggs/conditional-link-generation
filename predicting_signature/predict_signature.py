from torch.utils.data import DataLoader
import torch
import argparse
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn')
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--preprocessing', type=str, default=None)
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--kernel_size', type=int, default=2)
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