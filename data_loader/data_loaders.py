from jax import jit, random
import jax.numpy as jnp
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from base import BaseDataLoader

class FlattenAndCast(object):
    def __call__(self, pic):
        return np.ravel(np.array(pic, dtype=jnp.float32))

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self._dataset = dataset

    def __getitem__(self, idx):
        (data, target) = self._dataset[idx]
        return (data, target, np.array([idx]))

    def __len__(self):
        return len(self._dataset)

class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        self.data_dir = data_dir
        self.dataset = IndexedDataset(datasets.MNIST(self.data_dir, train=training, download=True, transform=FlattenAndCast()))
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

@jit
def binarize(rng_key, batch):
    return random.bernoulli(rng_key, batch).astype(batch.dtype)
