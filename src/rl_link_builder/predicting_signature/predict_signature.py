from torch.utils.data import DataLoader
import torch
import argparse
import lightning as pl
from lightning import Trainer
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from utils import *
from models.curiousity_models import MLP, CNN, TransformerEncoder, Reformer, GNN
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    # options: mlp, cnn, transformer_encoder, reformer, gnn
    parser.add_argument('--model', type=str, default=None)
    # options: clip (clip then normalize), log (abs log then scale) (only applicable to cnn and mlp)
    # other option: remove_cancelations (only applicable to transformer_encoder, reformer, and gnn)
    parser.add_argument('--preprocessing', type=str, default=None)
    # whether to treat each distinct signature as a separate category or as a continuous value 
    parser.add_argument('--classification', type=bool, default=False)
    parser.add_argument('--accelerator', type=str, default='cuda')
    # only applicable to mlp
    parser.add_argument('--hidden_size', type=int, default=100)
    parser.add_argument('--dropout', type=float, default=0.3)
    # only applicable to cnn
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--layer_norm', type=bool, default=True)
    # only applicable to transformer encoder and reformer
    parser.add_argument('--d_model', type=int, default=50)
    parser.add_argument('--nheads', type=int, default=2)
    # only applicable to transformer encoder, reformer, and gnn
    parser.add_argument('--num_layers', type=int, default=None)
    # only applicable to gnn
    parser.add_argument('--ohe_inverses', type=bool, default=False)
    return parser.parse_args()

def main():
    args = parse_args()

    # load the targets
    train_targets = np.load('y_train.npy')
    val_targets = np.load('y_val.npy')
    test_targets = np.load('y_test.npy')

    # load the LK data or braid words
    if args.preprocessing == 'clip' :
        train_data = np.load('clip_then_normalize_train.npy')
        val_data = np.load('clip_then_normalize_val.npy')
        test_data = np.load('clip_then_normalize_test.npy')
    elif args.preprocessing == 'log' :
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
        train_loader = get_graph_dataloader(train_braids, train_targets, 
                                            ohe_inverses=args.ohe_inverses, 
                                            batch_size=1, shuffle=True)
        val_loader = get_graph_dataloader(val_braids, val_targets, 
                                          ohe_inverses=args.ohe_inverses, 
                                          batch_size=64, shuffle=False)
        test_loader = get_graph_dataloader(test_braids, test_targets, 
                                           ohe_inverses=args.ohe_inverses, 
                                           batch_size=64, shuffle=False)

    # make model 
    num_generators = 12
    if args.model == 'mlp' :
        model = MLP(hidden_size=args.hidden_size, dropout=args.dropout, 
                    classification=args.classification)
    elif args.model == 'cnn' :
        model = CNN(kernel_size=args.kernel_size, layer_norm=args.layer_norm,
                    classification=args.classification)
    elif args.model == 'transformer_encoder' :
        model = TransformerEncoder(vocab_size=num_generators+1, d_model=args.d_model, 
                                   nhead=args.nhead, num_encoder_layers=args.num_layers, 
                                   dim_feedforward=1000, max_seq_length=45, 
                                   classification=args.classification)
    elif args.model == 'reformer' :
        model = Reformer(vocab_size=num_generators+1, d_model=args.d_model, 
                         nhead=args.nhead, num_layers=args.num_layers, max_seq_len=45, 
                         classification=args.classification)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor="val_l1_loss")

    

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # devices=torch.cuda.device_count(),
        max_epochs=100,
        callbacks=[lr_monitor, checkpoint_callback],
        # fast_dev_run=2, # for when testing
        enable_checkpointing=True, # so it returns the best model
        logger=pl.pytorch.loggers.TensorBoardLogger('logs/', name=experiment_name) 
        # max_time = "00:12:00:00",
        # num_nodes = args.num_nodes,
    )
 
    best_model = trainer.fit(model, train_dataloader, test_dataloader)
    trainer.test(best_model, val_dataloader)

    print(vars(args))