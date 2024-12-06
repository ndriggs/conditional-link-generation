import torch
from link_generation.potholders.utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from link_generation.vae.vae import VAE
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
import lightning as pl
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--potholder_size', type=int)
    parser.add_argument('--k', type=int, default=2)
    parser.add_argument('--accelerator', type=str, default='cuda')
    return parser.parse_args()

def main() :
    args = parse_args()
    state = torch.randint(low=0, high=2, size=(100000,args.potholder_size**2 - 2), dtype=torch.float32).to('cuda')

    P = state_to_potholder_pytorch(state)
    G = potholder_to_goeritz_pytorch(P)
    sig, log_det = goeritz_to_invariants(G)

    invariants = torch.cat([sig.unsqueeze(1),log_det.unsqueeze(1)], dim=1)

    X_train, X_val, y_train, y_val = train_test_split(state, invariants, test_size=0.2, random_state=15)
    
    scaler = StandardScaler()
    y_train_scaled = torch.tensor(scaler.fit_transform(y_train.cpu())).unsqueeze(1)
    y_val_scaled = torch.tensor(scaler.transform(y_val.cpu())).unsqueeze(1)

    train_loader = get_potholder_graph_data_loader(X_train, y_train_scaled, args.potholder_size, args.k, 128, True)
    val_loader = get_potholder_graph_data_loader(X_val, y_val_scaled, args.potholder_size, args.k, 128, False)

    model = VAE(mean=scaler.mean_, scale=scaler.scale_, potholder_size=args.potholder_size)

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    checkpoint_callback = ModelCheckpoint(monitor="val_loss")

    experiment_name = f'pot2pot_vae_{args.potholder_size}_k_{args.k}'

    trainer = pl.Trainer(
        accelerator=args.accelerator,
        # devices=torch.cuda.device_count(),
        max_epochs=100,
        callbacks=[lr_monitor, checkpoint_callback],
        # fast_dev_run=2, # for when testing
        enable_checkpointing=True, # so it returns the best model
        logger=pl.pytorch.loggers.TensorBoardLogger('src/link_generation/vae/logs/', name=experiment_name) 
        # max_time = "00:12:00:00",
        # num_nodes = args.num_nodes,
    )
 
    best_model = trainer.fit(model, train_loader, val_loader)
    trainer.test(best_model, val_loader, ckpt_path='best')

    print(vars(args))

if __name__=='__main__' :
    main()