import jax.numpy as jnp
from sklearn.metrics import pairwise_kernels


def kernel_matrix(x, y, gamma):
    """Compute the kernel matrix."""
    return pairwise_kernels(x, y, metric="rbf", gamma=gamma)


def calculate_kid(x, y):
    """Calculate the kernel inception distance."""

    # Set the kernel width as 1/num_features
    gamma = 1.0 / x.shape[-1]

    # Compute the kernel matrices
    k_xx = kernel_matrix(x, x, gamma)
    k_yy = kernel_matrix(y, y, gamma)
    k_xy = kernel_matrix(x, y, gamma)

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
