from torch import Tensor
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from src.data.datamodule import MolecularDataModule
from src.data.featurization import LigandSimpleFeaturization
from src.models import MoleculeBinaryClassifier
from src.models.enum import Architecture, Parametrization, DensityType


def get_datamodule(dataset_name: str, batch_size: int = 32):
    """Create dataset with given name, e.g. AMES"""
    dataset = MolecularDataModule(
        f"data/{dataset_name}/",
        pre_transform=LigandSimpleFeaturization(),
        pre_filter=None,
        batch_size=batch_size,
        val_batch_size=32,
        test_batch_size=32,
    )
    return dataset


def get_logger(run, **kwargs):
    return WandbLogger(log_model="all", experiment=run, **kwargs)


def get_callbacks():
    val_checkpoint = ModelCheckpoint(
        filename="epoch={epoch}-step={step}-val_loss={loss/val:.3f}",
        monitor="loss/val",
        mode="min",
        auto_insert_metric_name=False,
    )
    latest_checkpoint = ModelCheckpoint(
        filename="latest-{epoch}-{step}",
        monitor="epoch",
        mode="max",
        every_n_epochs=1,
        save_last=True,
    )
    early_stop_callback = EarlyStopping(
    monitor="loss/val",  
    min_delta=0.00,     
    patience=50,         
    verbose=False,       
    mode="min",
    )
    return [val_checkpoint, latest_checkpoint, early_stop_callback]


def get_model(
    lr=1e-4,
    architecture=Architecture.AttFpPost,
    density_type=DensityType.IAF,
    n_ffn_layers=3,
    latent_dim=6,
    n_density=6,
    hidden_features=256,
    num_layers=3,
    num_timesteps=2,
    dropout=0.2,
    N=Tensor([100, 100]),
):
    return MoleculeBinaryClassifier(
        lr=lr,
        clip_grad=True,
        architecture=architecture,
        density_type=density_type,
        n_ffn_layers=n_ffn_layers,
        latent_dim=latent_dim,
        n_density=n_density,
        hidden_features=hidden_features,
        num_layers=num_layers,
        num_timesteps=num_timesteps,
        dropout=dropout,
        N=N,
    )
