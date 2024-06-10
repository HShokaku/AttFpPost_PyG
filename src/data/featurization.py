import torch
import pandas as pd
from torch_geometric.data import Data
from torch_geometric.utils import from_smiles


class LigandSimpleFeaturization:
    def __init__(self, **kargs) -> None:
        self.kargs = kargs

    def __call__(self, row: pd.Series) -> Data:
        data = from_smiles(row["smiles"])
        data["target"] = torch.tensor(row["Y"], dtype=torch.long)

        return data

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}"
