import argparse
from types import SimpleNamespace

import pytorch_lightning as pl
import torch
torch.set_float32_matmul_precision('high')

import wandb
from _util import get_callbacks, get_datamodule, get_logger, get_model
from src.models.enum import Architecture, DensityType
from src.util import disable_rdkit_logging


def str_to_bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def train(config, accelerator="gpu" if torch.cuda.is_available() else None, devices=[1]):
    run = wandb.init(
        project="AttFp-PyG",
        name="debug",
    )
    pl.seed_everything(config.seed)

    data_module = get_datamodule(
        config.dataset_name, batch_size=config.batch_size // len(devices)
    )

    model = get_model(
        lr=config.lr,
        architecture=config.architecture,
        density_type=config.density_type,
        n_ffn_layers=config.n_ffn_layers,
        latent_dim=config.latent_dim,
        n_density=config.n_density,
        hidden_features=config.hidden_features,
        num_layers=config.num_layers,
        num_timesteps=config.num_timesteps,
        dropout=config.dropout,
        N=data_module.get_train_labels_ratio(),
    )

    model.setup_metrics()

    wandb_logger = get_logger(run)
    wandb_logger.watch(model)

    callbacks = get_callbacks()
    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator=accelerator,
        devices=devices,
        num_sanity_val_steps=0,
        logger=wandb_logger,
        callbacks=callbacks,
    )
    trainer.fit(model, data_module)

def parse_args():
    default_config = SimpleNamespace(
        seed=1,
        num_epochs=500,
        batch_size=16,
        lr=5e-4,
        dataset_name="debug",
        architecture=Architecture.AttFpPost,
        density_type=DensityType.IAF,
        n_ffn_layers=3,
        latent_dim=6,
        n_density=6,
        hidden_features=256,
        num_layers=3,
        num_timesteps=2,
        dropout=0.2,
    )

    parser = argparse.ArgumentParser(
        prog="train_model.py",
        description="Train model",
        epilog="Example: python train_model.py",
    )
    parser.add_argument(
        "--architecture",
        type=Architecture,
        help="Architecture",
        default=default_config.architecture,
    )
    parser.add_argument(
        "--density_type",
        type=DensityType,
        help="DensityType",
        default=default_config.density_type,
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed",
        default=default_config.seed,
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        help="Dataset name",
        default=default_config.dataset_name,
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        help="Number of epochs",
        default=default_config.num_epochs,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=default_config.batch_size,
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="Learning rate",
        default=default_config.lr,
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        help="Number of layers",
        default=default_config.num_layers,
    )
    parser.add_argument(
        "--hidden_features",
        type=int,
        help="Number of hidden features",
        default=default_config.hidden_features,
    )
    parser.add_argument(
        "--n_ffn_layers",
        type=int,
        help="Number of ffn layers",
        default=default_config.n_ffn_layers,
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        help="Number of latent dimensions",
        default=default_config.n_ffn_layers,
    )
    parser.add_argument(
        "--n_density",
        type=int,
        help="Number of density",
        default=default_config.n_density,
    )
    parser.add_argument(
        "--num_timesteps",
        type=int,
        help="Number of timestep",
        default=default_config.num_timesteps,
    )
    parser.add_argument(
        "--dropout",
        type=float,
        help="Dropout rate",
        default=default_config.dropout,
    )

    config = parser.parse_args()

    return config


def main():
    disable_rdkit_logging()
    config = parse_args()
    train(config)


if __name__ == "__main__":
    main()
