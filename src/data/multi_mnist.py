import jax.numpy as jnp
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, TensorDataset
from torchvision import datasets, transforms
from typing import Callable, Optional, Tuple

from . import datamodule

def imresize(arr, size):
    return np.array(Image.fromarray(arr).resize(size))

def sample_one(canvas_side, mnist):
    i = np.random.randint(len(mnist))
    digit, label = mnist[i]
    scale = 0.1 * np.random.randn() + 1.3
    new_size = tuple(int(s / scale) for s in digit.shape)
    resized = imresize(digit, new_size)
    w = resized.shape[0]
    assert w == resized.shape[1]
    padding = canvas_side - w
    pad_l = np.random.randint(0, padding)
    pad_r = np.random.randint(0, padding)
    pad_width = ((pad_l, padding - pad_l), (pad_r, padding - pad_r))
    positioned = np.pad(resized, pad_width, "constant", constant_values=0)
    return positioned, label


def sample_multimnist(num_digits, canvas_side, mnist):
    canvas = np.zeros((canvas_side, canvas_side))
    labels = []
    for _ in range(num_digits):
        positioned_digit, label = sample_one(canvas_side, mnist)
        canvas += positioned_digit
        labels.append(label)
    # Crude check for overlapping digits.
    if np.max(canvas) > 255:
        return sample_multimnist(num_digits, canvas_side, mnist)
    else:
        return canvas, labels

def construct_multimnist(mnist, num_digits, canvas_side, file_path):
    x, y = [], []
    for _ in range(len(mnist)):
        canvas, labels = sample_multimnist(num_digits, canvas_side, mnist)
        x.append(canvas)
        y.append(labels)
    return np.array(x, dtype=np.uint8), np.array(y, dtype=np.uint8)

def load_multimnist(data_dir, canvas_side=50, download=True, num_digits=2,
                    train=True):
    trainval = "train" if train else "test"
    file_path = os.path.join(
        data_dir,
        "multi_mnist_{}_{}_{}_uint8.npz".format(num_digits, canvas_side,
                                                trainval)
    )
    mnist = datasets.MNIST(data_dir, train=train, download=download,
                           transform=transforms.Lambda(
                               lambda x: np.array(x, dtype=jnp.uint8)
                           ))
    if os.path.exists(file_path):
        data = np.load(file_path, allow_pickle=True)
        digits, targets = data["digits"], data["targets"]
        if digits.shape[0] != len(mnist) or targets.shape[0] != len(mnist):
            digits, targets = construct_multimnist(mnist, num_digits,
                                                   canvas_side, file_path)
    else:
        # Set RNG to known state.
        rng_state = np.random.get_state()
        np.random.seed(681307)
        digits, targets = construct_multimnist(mnist, num_digits,
                                               canvas_side, file_path)
        # Revert RNG state.
        np.random.set_state(rng_state)
        with open(file_path, "wb") as f:
            np.savez_compressed(f, digits=digits, targets=targets)
    return TensorDataset(torch.tensor(digits), torch.tensor(targets))

class MultiMnist(datasets.VisionDataset):
    def __init__(self, root, canvas_side=50, download=True, num_digits=2,
                 train=True, target_transform: Optional[Callable]=None,
                 transform: Optional[Callable]=None):
        self.tensors = load_multimnist(root, canvas_side=canvas_side,
                                       download=download, num_digits=num_digits,
                                       train=train)

        super().__init__(root, target_transform=target_transform,
                         transform=transform)

    def __getitem__(self, index):
        from torchvision.utils import _Image_fromarray

        img, target = self.tensors[index]
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = _Image_fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.tensors)

class MultiMnistDataModule(datamodule.DataModule):
    def __init__(self, canvas_side=50, num_digits=2, **kwargs):
        self._canvas_side = canvas_side
        self._num_digits = num_digits
        self.num_data = 0
        self.transforms = transforms.Compose([
            transforms.Lambda(
                lambda x: np.array(x, dtype=jnp.float32) / 255.
            ),
            transforms.Lambda(lambda x: x.reshape(1, canvas_side, canvas_side)),
        ])
        super().__init__(**kwargs)

    @property
    def canvas_side(self):
        return self._canvas_side

    @property
    def num_digits(self):
        return self._num_digits

    def prepare_data(self) -> Tuple[Dataset, Dataset]:
        data_train = MultiMnist(self.data_dir, canvas_side=self.canvas_side,
                                download=True, num_digits=self.num_digits,
                                train=True, transform=self.transforms)
        data_test = MultiMnist(self.data_dir, canvas_side=self.canvas_side,
                                download=True, num_digits=self.num_digits,
                                train=False, transform=self.transforms)
        self.num_data += len(data_train) + len(data_test)
        return data_train, data_test

    @property
    def shape(self) -> Tuple:
        return (1, self._canvas_side, self._canvas_side)
