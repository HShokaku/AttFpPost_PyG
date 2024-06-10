import torch
from torch import Tensor
from torch import nn as nn
from torch.nn import functional as F
from torch.distributions.dirichlet import Dirichlet
from torch_geometric.data import Data as DataBatch
from torch_geometric.nn.models import AttentiveFP

from src.models.attfppost.density import NormalizingFlowDensity
from src.models.enum import DensityType


class AttentiveFpPost(nn.Module):
    """
    Attentive FP with Posterior Network.
    """
    def __init__(
        self,
        # number of samples per class: torch.Tensor([N_0, N_1])
        N: Tensor,
        # Parameters for Attentive FP Encoder
        num_atom_feats: int = 9,
        num_bond_feats: int = 3,
        hidden_features: int = 256,
        num_layers: int = 3,
        num_timesteps: int = 2,
        dropout: float = 0.1,
        n_ffn_layers: int = 3,
        # Parameters for the Normalizing Flow
        latent_dim: int = 6,
        n_density: int = 6,
        density_type=DensityType.IAF
    ) -> None:
        super().__init__()
        self.N = N
        # create encoder
        self.encoder = AttentiveFP(
            in_channels=num_atom_feats,
            hidden_channels=hidden_features,
            out_channels=hidden_features,
            edge_dim=num_bond_feats,
            num_layers=num_layers,
            num_timesteps=num_timesteps,
            dropout=dropout,
        )
        # create feed-forward network
        ffn = [nn.Linear(hidden_features, hidden_features)]
        for _ in range(n_ffn_layers - 2):
            ffn.extend([
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_features, hidden_features),
            ])
        ffn.extend([
            nn.ReLU(), 
            nn.Dropout(dropout), 
            nn.Linear(hidden_features, latent_dim)
        ])
        self.ffn = nn.Sequential(*ffn)
        # create Normalizing Flow
        self.density_estimation = nn.ModuleList(
            [
                NormalizingFlowDensity(latent_dim, n_density, density_type),
                NormalizingFlowDensity(latent_dim, n_density, density_type),
            ]
        )
        self.batch_norm = nn.BatchNorm1d(latent_dim)
        self.reset_parameters()

    def reset_parameters(self):
        self.encoder.reset_parameters()
        for module in self.ffn:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        self.batch_norm.reset_parameters()

    def forward(self, x: DataBatch, targets: Tensor):
        alpha, preds = self._forward(x)
        targets_hot = F.one_hot(targets, num_classes=2)
        loss = self.UCE_loss(alpha, targets_hot)
        return loss, preds[:, 1] 
    
    @torch.no_grad()
    def predict(self, x: DataBatch):
        alpha, preds = self._forward(x)
        return alpha, preds[:, 1] 

    def _forward(self, x: DataBatch):
        device = x.x.device
        z = self.encoder(
            x.x.float(), 
            x.edge_index, 
            x.edge_attr, 
            x.batch
        )
        zk = self.ffn(z)
        zk = self.batch_norm(zk)

        log_q_zk = torch.zeros((zk.shape[0], 2), device=device)
        alpha = torch.zeros((zk.shape[0], 2), device=device)

        if isinstance(self.density_estimation, nn.ModuleList):
            for cls in range(2):
                log_p = self.density_estimation[cls].log_prob(zk)
                log_q_zk[:, cls] = log_p
                alpha[:, cls] = 1. + (self.N[cls] * torch.exp(log_q_zk[:, cls]))
        else:
            log_q_zk = self.density_estimation.log_prob(zk)
            alpha = 1. + (self.N[:, None] * torch.exp(log_q_zk)).permute(1, 0)

        soft_output_pred = torch.nn.functional.normalize(alpha, p=1)

        return alpha, soft_output_pred
    
    def UCE_loss(self, alpha, targets_hot):
        alpha_0 = alpha.sum(1).unsqueeze(-1).repeat(1, 2)
        entropy_reg = Dirichlet(alpha).entropy()
        
        return torch.sum(targets_hot * (torch.digamma(alpha_0) - torch.digamma(alpha))) - 1e-5 * torch.sum(entropy_reg)
