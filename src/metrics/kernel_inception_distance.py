from sklearn.metrics import pairwise_kernels
import jax.numpy as jnp


def kernel_matrix(x, y, gamma):
    """Compute the kernel matrix."""
    return pairwise_kernels(x, y, metric="rbf", gamma=gamma)


def calculate_kid(x, y):
    """
    Calculate the kernel inception distance.
    
    
    https://doi.org/10.48550/arXiv.1801.01401
    """

    # Set the kernel width as 1/num_features
    gamma = 1.0 / x.shape[-1]

    # Compute the kernel matrices
    k_xx = kernel_matrix(x, x, gamma) - jnp.diag(jnp.ones(x.shape[0]))
    k_yy = kernel_matrix(y, y, gamma) - jnp.diag(jnp.ones(x.shape[0]))
    k_xy = kernel_matrix(x, y, gamma) - jnp.diag(jnp.ones(x.shape[0]))

    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
