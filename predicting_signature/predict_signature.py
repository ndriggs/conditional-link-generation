from torch.utils.data import DataLoader
import torch
import argparse
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import load_braid_words, remove_cancelations, BraidDataset, pad_braid_words
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # options: mlp, cnn, transformer_encoder, reformer, gnn
    parser.add_argument('--model', type=str, default=None)
    # only applicable to mlp
    parser.add_argument('--hidden_size', type=int, default=100)
    # only applicable to mlp
    parser.add_argument('--dropout', type=float, default=0.3)
    # options: clip_then_normalize, abs_log_then_scale (only applicable to cnn and mlp)
    # other option: remove_cancelations (only applicable to transformer_encoder, reformer, and gnn)
    parser.add_argument('--preprocessing', type=str, default=None)
    # whether to treat each distinct signature as a separate category or as a continuous value 
    # applicable to all models
    parser.add_argument('--classification', type=bool, default=False)
    parser.add_argument('--accelerator', type=str, default='gpu')
    # options: 2 or 3 (only applicable to cnn)
    parser.add_argument('--kernel_size', type=int, default=2)
    # only applicable to cnn
    parser.add_argument('--layer_norm', type=bool, default=True)
    # only applicable to gnn, options: 2, 3 or 4
    parser.add_argument('--num_gnn_layers', type=int, default=3)
    return parser.parse_args()

def main():
    args = parse_args()

    # load the targets
    train_targets = np.load('y_train.npy')
    val_targets = np.load('y_val.npy')
    test_targets = np.load('y_test.npy')

    # load the LK data or braid words
    if args.preprocessing == 'clip_then_normalize' :
        train_data = np.load('clip_then_normalize_train.npy')
        val_data = np.load('clip_then_normalize_val.npy')
        test_data = np.load('clip_then_normalize_test.npy')
    elif args.preprocessing == 'abs_log_then_scale' :
        train_data = np.load('train_log_scaled.npy')
        val_data = np.load('val_log_scaled.npy')
        test_data = np.load('test_log_scaled.npy')
    elif args.preprocessing == 'remove_cancelations' :
        train_braids = remove_cancelations('train')
        val_braids = remove_cancelations('val')
        test_braids = remove_cancelations('test')
    elif args.model in ['transformer_encoder', 'reformer', 'gnn'] :
        train_braids = load_braid_words('train')
        val_braids = load_braid_words('val')
        test_braids = load_braid_words('test')

    # create the datasets for all models but GNN
    if args.model in ['mlp', 'cnn'] :
        train_dataset = BraidDataset(data=train_data, targets=train_targets, 
                                     classification=args.classification,
                                     cnn=(args.model == 'cnn'))
        val_dataset = BraidDataset(data=val_data, targets=val_targets, 
                                   classification=args.classification,
                                   cnn=(args.model == 'cnn'))
        test_dataset = BraidDataset(data=test_data, targets=test_targets, 
                                    classification=args.classification,
                                    cnn=(args.model == 'cnn'))

    elif args.model in ['transformer_encoder', 'reformer'] :
        train_padded, train_lengths = pad_braid_words(train_braids)
        val_padded, val_lengths = pad_braid_words(val_braids)
        test_padded, test_lengths = pad_braid_words(test_braids)

        train_dataset = BraidDataset(data=train_padded, targets=train_targets, 
                                     classification=args.classification,
                                     seq_lengths=train_lengths)
        val_dataset = BraidDataset(data=val_padded, targets=val_targets, 
                                   classification=args.classification,
                                   seq_lengths=val_lengths)
        test_dataset = BraidDataset(data=test_padded, targets=test_targets, 
                                    classification=args.classification,
                                    seq_lengths=test_lengths)

    # create the dataloaders for all models but GNN
    if args.model in ['mlp', 'cnn', 'transformer_encoder', 'reformer'] : 
        train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # create the datasets and dataloaders for GNN
    if args.model == 'gnn' :

        