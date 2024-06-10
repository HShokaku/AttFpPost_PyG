import sys
import multiprocessing as mp
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

import torch
import pandas as pd
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map


class MolecularDataset(InMemoryDataset):
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        log: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        super().__init__(root, transform, pre_transform, pre_filter, log=log)
        split_file = Path(self.processed_dir) / f"{split}.pt"
        self.data, self.slices = torch.load(split_file)

    @property
    def raw_file_names(self) -> List[str]:
        return [
            "train.csv",
            "val.csv",
            "test.csv",
        ]
    
    @property
    def processed_file_names(self) -> Union[str, List[str], Tuple]:
        return [f"{self.split}.pt"]
    
    def _load_splits(self, split: str = None):
        return pd.read_csv(Path(self.raw_dir) / f"{split}.csv")

    def process(self):
        processed_dir = Path(self.processed_dir)
        processed_dir.mkdir(exist_ok=True, parents=True)

        self._featurize_split(max_workers=0)

    def _featurize_split(self, split: str = None, max_workers: int = None):
        if split is None:
            split = self.split

        processed_dir = Path(self.processed_dir)
        split_file = processed_dir / f"{split}.pt"

        if split_file.exists():
            self._log(f"Skipping {split} as {split_file} already exists")
            return
        
        df = self._load_splits(split)
        self._log(
            f"Processing {len(df)} complexes to {split_file} for {split}"
        )

        if max_workers is None or max_workers == 0:
            self._log("Transforming data with single process")
            transformed_data = []
            for row_tuple in (pbar := tqdm(df.iterrows(), total=len(df))):
                candidate = self._featurize_candidate(row_tuple)
                transformed_data.append(candidate)
                pbar.set_description(f"Transformed {len(transformed_data)} data")
        else:
            self._log(f"Transforming data with {max_workers} process")
            transformed_data = []
            pool = mp.Pool(processes=max_workers)
            with tqdm(total=len(df)) as pbar:
                for candidate in pool.imap(self._featurize_candidate, df.iterrows()):
                    transformed_data.append(candidate)
                    pbar.update(1)
            pool.close()
            pool.join()

        transformed_data = [c for c in transformed_data if c is not None]
        self._log(f"Transformed {len(transformed_data)} complexes")
        data, slices = self.collate(transformed_data)
        torch.save((data, slices), split_file)

    def _featurize_candidate(self, row_tuple) -> Optional[Data]:
        index, row = row_tuple
        try:
            if self.pre_transform is not None:
                data = self.pre_transform(row)
            return data
        
        except Exception as e:
            smiles = row["smiles"]
            self._log(f"Could not process {smiles} in {self.split}: {e}")
            return None

    def _log(self, msg: str):
        if self.log:
            tqdm.write(msg, file=sys.stderr)
