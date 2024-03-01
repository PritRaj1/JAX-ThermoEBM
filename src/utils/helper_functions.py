from torchvision import datasets, transforms
import jax
import numpy
import jax.numpy as jnp
from torch.utils import data
from jax.tree_util import tree_map
import numpy as np


def numpy_collate(batch):
    return tree_map(np.asarray, data.default_collate(batch))


class NumpyLoader(data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super(self.__class__, self).__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=numpy_collate,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
        )


def get_data(name):

    img_dim = 64 if name == "CelebA" else 32

    class Resize_Normalise(object):
        def __call__(self, pic):
            # return np.ravel(np.array(pic, dtype=jnp.float32))

            # Resize
            pic = pic.resize((img_dim, img_dim))

            # Normalise
            return (np.array(pic, dtype=jnp.float32) - 127.5) / 127.5            

    if name == "CIFAR10":
        data = {
            "train": datasets.CIFAR10(root="dataset/", train=True, download=True, transform=Resize_Normalise()),
            "test": datasets.CIFAR10(root="dataset/", train=False, download=True, transform=Resize_Normalise()),
        }
    elif name == "SVHN":
        data = {
            "train": datasets.SVHN(root="dataset/", split="train", download=True, transform=Resize_Normalise()),
            "test": datasets.SVHN(root="dataset/", split="test", download=True, transform=Resize_Normalise()),
        }
    elif name == "CelebA":
        data = {
            "train": datasets.CelebA(root="dataset/", split="train", download=True, transform=Resize_Normalise()),
            "test": datasets.CelebA(root="dataset/", split="test", download=True, transform=Resize_Normalise()),
        }
        raise ValueError("Invalid dataset name.")

    return data["train"], data["test"], img_dim


def get_grad_var(grad_ebm, grad_gen):

    # Get gradients from grad dictionaries
    grad_ebm = jax.tree_util.tree_flatten(grad_ebm)[0]
    grad_gen = jax.tree_util.tree_flatten(grad_gen)[0]

    # Flatten the gradients
    grad_ebm = jnp.concatenate([jnp.ravel(g) for g in grad_ebm])
    grad_gen = jnp.concatenate([jnp.ravel(g) for g in grad_gen])

    return jnp.var(jnp.concatenate([grad_ebm, grad_gen]))
