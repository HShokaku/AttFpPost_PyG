from collections import deque
from typing import Dict, Optional, Union

import numpy as np
import pytorch_lightning as pl
import torch
import torch.optim as optim
from rdkit.Chem import Draw
from torch import Tensor
from torch_geometric.data import Data
from torchmetrics import Accuracy, AUROC, AveragePrecision

from src.models.attfppost.model import AttentiveFP, AttentiveFpPost
from src.models.enum import Architecture, Parametrization, DensityType
from src.models.util import skip_computation_on_oom


class MoleculeBinaryClassifier(pl.LightningModule):
    def __init__(
        self,
        # Training parameters
        lr=1e-4,
        clip_grad=False,
        # Model parameters
        architecture: Architecture = Architecture.AttFpPost,
        density_type: DensityType = DensityType.IAF,
        n_ffn_layers=3,
        latent_dim=6,
        n_density=6,
        # Encoder parameters
        hidden_features=256,
        num_layers=3,
        num_timesteps=2,
        dropout=0.2,
        # Dataset parameters
        N: Tensor = Tensor([1, 1]),
        atom_features=9,
        bond_features=3,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        encoder_params = dict(
            num_atom_feats=atom_features,
            num_bond_feats=bond_features,
            hidden_features=hidden_features,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )

        if architecture == Architecture.AttFp:
            self.model = AttentiveFP(
                **encoder_params
            )
        elif architecture == Architecture.AttFpPost:
            self.model = AttentiveFpPost(
                N=N,
                n_ffn_layers=n_ffn_layers,
                latent_dim=latent_dim,
                n_density=n_density,
                density_type=density_type,
                **encoder_params,
            )
        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        self.lr = lr
        self.clip_grad = clip_grad
        if self.clip_grad:
            self.gradient_norm_queue = deque([3000.0], maxlen=50)
        self.validation_metrics = None

    def setup_metrics(self):
        self.validation_metrics = torch.nn.ModuleDict(
            {
                "Accuracy": Accuracy(task="binary"),
                "AUROC": AUROC(task="binary"),
                "AveragePrecision": AveragePrecision(task="binary"),
            }
        )

    @skip_computation_on_oom(
        return_value=None, error_message="Skipping batch due to OOM"
    )
    def training_step(self, batch, batch_idx):
        (
            loss,
            preds,
        ) = self.model(batch, batch.target)
        self.log("loss/train", loss, batch_size=batch.num_graphs)
        return loss

    def validation_step(self, batch, batch_idx):
        (
            loss,
            preds,
        ) = self.model(batch, batch.target)
        
        self.log("loss/val", loss, batch_size=batch.num_graphs, sync_dist=True)
        
        for k, metric in self.validation_metrics.items():
            metric(preds, batch.target)
            self.log(
                f"{k}/val",
                metric,
                batch_size=batch.num_graphs,
                sync_dist=True,
            )

        return loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.lr, amsgrad=True, weight_decay=1e-12
        )
        return optimizer

    def configure_gradient_clipping(
        self,
        optimizer: optim.Optimizer,
        # optimizer_idx: int,
        gradient_clip_val: Optional[Union[int, float]] = None,
        gradient_clip_algorithm: Optional[str] = None,
    ) -> None:
        if not self.clip_grad:
            return

        # Allow gradient norm to be 150% + 2 * (standard deviation of the recent history).
        max_grad_norm: float = 1.5 * np.mean(self.gradient_norm_queue) + 2 * np.std(
            self.gradient_norm_queue
        )

        # Get current grad_norm
        grad_norm = float(get_grad_norm(optimizer))

        self.gradient_norm_queue.append(min(grad_norm, max_grad_norm))

        self.clip_gradients(
            optimizer, gradient_clip_val=max_grad_norm, gradient_clip_algorithm="norm"
        )

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        """Override this method to change the default behaviour of ``log_grad_norm``.

        If clipping gradients, the gradients will not have been clipped yet.

        Args:
            grad_norm_dict: Dictionary containing current grad norm metrics

        Example::

            # DEFAULT
            def log_grad_norm(self, grad_norm_dict):
                self.log_dict(grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        """
        results = self.trainer._results
        if isinstance(results.batch, Data):
            results.batch_size = results.batch.num_graphs
        self.log_dict(
            grad_norm_dict, on_step=True, on_epoch=True, prog_bar=False, logger=True
        )


def get_grad_norm(
    optimizer: torch.optim.Optimizer, norm_type: float = 2.0
) -> torch.Tensor:
    """
    Adapted from: https://pytorch.org/docs/stable/_modules/torch/nn/utils/clip_grad.html#clip_grad_norm_
    """
    parameters = [p for g in optimizer.param_groups for p in g["params"]]
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]

    norm_type = float(norm_type)

    if len(parameters) == 0:
        return torch.tensor(0.0)

    device = parameters[0].grad.device

    total_norm = torch.norm(
        torch.stack(
            [torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]
        ),
        norm_type,
    )

    return total_norm
