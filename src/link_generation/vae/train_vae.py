from link_generation.vae.vae import VAE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from link_generation.vae.utils import get_knot_braids_sig_and_det
from link_generation.predicting_signature.utils import get_knot_graph_dataloader
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import lightning as pl
import argparse
import torch


def parse_args() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--potholder_size', type=int, default=9)
    parser.add_argument('--accelerator', type=str, default='cuda')
    return parser.parse_args()

def main() :
    args = parse_args()

    # load the data, but only for knots 
    train_braids, train_sig, train_log_det = get_knot_braids_sig_and_det('train')
    val_braids, val_sig, val_log_det = get_knot_braids_sig_and_det('val')

    # scale the signature 
    sig_scaler = MinMaxScaler()
    sig_scaler.fit(train_sig.reshape(-1, 1))
    scaled_train_sig = sig_scaler.transform(train_sig.reshape(-1, 1))
    scaled_val_sig = sig_scaler.transform(val_sig.reshape(-1, 1))

    # scale the determinant
    log_det_scaler = StandardScaler()
    log_det_scaler.fit(train_log_det.reshape(-1, 1))
    scaled_train_log_det = log_det_scaler.transform(train_log_det.reshape(-1, 1))
    scaled_val_log_det = log_det_scaler.transform(val_log_det.reshape(-1, 1))

    # concat the invariants
    scaled_invariants_train = torch.cat([torch.tensor(scaled_train_sig, dtype=torch.float32).unsqueeze(1),
                                         torch.tensor(scaled_train_log_det, dtype=torch.float32).unsqueeze(1)], 
                                         dim=1).unsqueeze(1)
    scaled_invariants_val = torch.cat([torch.tensor(scaled_val_sig, dtype=torch.float32).unsqueeze(1),
                                       torch.tensor(scaled_val_log_det, dtype=torch.float32).unsqueeze(1)], 
                                       dim=1).unsqueeze(1)
    
    # get the data loaders
    train_loader = get_knot_graph_dataloader(train_braids, scaled_invariants_train, both=False,
                                             pos_neg=False, ohe_inverses=True, undirected=True, 
                                             laplacian=False, k=1, batch_size=64, shuffle=True)
    val_loader = get_knot_graph_dataloader(val_braids, scaled_invariants_val, both=False,
                                           pos_neg=False, ohe_inverses=True, undirected=True, 
                                           laplacian=False, k=1, batch_size=64, shuffle=False)
    
    # initialize the model
    model = VAE(log_det_mean=log_det_scaler.mean_[0], log_det_std=log_det_scaler.scale_[0],
                sig_min=sig_scaler.data_min_[0], sig_max=sig_scaler.data_max_[0], 
                potholder_size=args.potholder_size)
    
    # set up the pytorch lightning monitors and callbacks
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # devices=torch.cuda.device_count(),
        max_epochs=100,
        callbacks=[lr_monitor, checkpoint_callback],
        # fast_dev_run=2, # for when testing
        enable_checkpointing=True, # so it returns the best model
        logger=pl.pytorch.loggers.TensorBoardLogger('src/link_generation/vae/logs/', name=f'vae_potholder_{args.potholder_size}') 
        # max_time = "00:12:00:00",
        # num_nodes = args.num_nodes,
    )
 
    best_model = trainer.fit(model, train_loader, val_loader)
    trainer.test(best_model, val_loader, ckpt_path='best')

    print(vars(args))

if __name__ == '__main__':
    main()
    







