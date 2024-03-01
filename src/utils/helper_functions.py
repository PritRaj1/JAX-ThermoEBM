from torchvision import datasets, transforms
import jax
import numpy
import jax.numpy as jnp


def get_data(name):

    img_dim = 64 if name == "CelebA" else 32

    import jax.numpy as jnp

    transform = transforms.Compose(
        [
            transforms.Resize(
                (img_dim, img_dim)
            ),  # Resize the images to defined resolution
            transforms.ToTensor(),
            transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
            ),  # Normalize the image tensors
        ]
    )

    if name == "CIFAR10":
        dataset = datasets.CIFAR10(
            root="dataset/", train=True, transform=transform, download=True
        )
        val_dataset = datasets.CIFAR10(
            root="dataset/", train=False, transform=transform, download=True
        )
    elif name == "SVHN":
        dataset = datasets.SVHN(
            root="dataset/", split="train", transform=transform, download=True
        )
        val_dataset = datasets.SVHN(
            root="dataset/", split="test", transform=transform, download=True
        )
    elif name == "CelebA":
        dataset = datasets.CelebA(
            root="dataset/", split="train", transform=transform, download=True
        )
        val_dataset = datasets.CelebA(
            root="dataset/", split="test", transform=transform, download=True
        )
    else:
        raise ValueError("Invalid dataset name.")

    return dataset, val_dataset, str(img_dim)

def get_grad_var(grad_ebm, grad_gen):
    
     # Get gradients from grad dictionaries
    grad_ebm = jax.tree_util.tree_flatten(grad_ebm)[0]
    grad_gen = jax.tree_util.tree_flatten(grad_gen)[0]   

    # Flatten the gradients
    grad_ebm = jnp.concatenate([jnp.ravel(g) for g in grad_ebm])
    grad_gen = jnp.concatenate([jnp.ravel(g) for g in grad_gen])

    return jnp.var(jnp.concatenate([grad_ebm, grad_gen]))
