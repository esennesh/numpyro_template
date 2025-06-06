from abc import abstractmethod, abstractproperty
import numpy as np
from jax.tree_util import tree_map
from torch.utils.data import Dataset, DataLoader, default_collate, random_split
from typing import Tuple

def numpy_collate(batch):
  return tree_map(np.asarray, default_collate(batch))

class DataModule:
    """
    Base class for all data modules
    """
    def __init__(self, batch_size: int=64, data_dir: str="data/",
                 collate_fn=numpy_collate, num_workers: int=1,
                 pin_memory: bool=False, shuffle: bool=True,
                 validation_split: float=0.1):
        self.data_dir = data_dir
        self.validation_split = validation_split
        self.data_train, self.data_val, self.data_test =\
            self.setup(*self.prepare_data(), validation_split)

        self.dataloader_kwargs = {
            'batch_size': batch_size,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'pin_memory': pin_memory,
            'shuffle': shuffle,
        }

    @abstractmethod
    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        raise NotImplementedError

    @staticmethod
    def setup(train_data, test_data, validation_split) -> Tuple[Dataset, Dataset, Dataset]:
        val_length = int(len(train_data) * validation_split)
        train_val_split = (len(train_data) - val_length, val_length)
        train_data, val_data = random_split(dataset=train_data,
                                            lengths=train_val_split)
        return train_data, val_data, test_data

    @abstractproperty
    def shape(self) -> Tuple:
        raise NotImplementedError

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_test,
            **self.dataloader_kwargs,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            **self.dataloader_kwargs,
        )

    def valid_dataloader(self) -> DataLoader:
        return DataLoader(
            dataset=self.data_train,
            **self.dataloader_kwargs,
        )
