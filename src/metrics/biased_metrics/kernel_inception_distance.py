
import jax.numpy as jnp
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])

def kernel_matrix(x, y, gamma):
    """Compute the rbf pairwise kernel matrix."""
    
    # Compute the pairwise squared Euclidean distances
    pairwise_distances_sq = jnp.sum((x[:, None] - y[None, :])**2, axis=-1)
    
    # Compute the kernel matrix
    return jnp.exp(-gamma * pairwise_distances_sq)

def calculate_kid(x, y):
    """
    Calculate the kernel inception distance.
    
    
    https://doi.org/10.48550/arXiv.1801.01401
    """

    # Set the kernel width as 1/num_features
    gamma = 1.0 / x.shape[-1]

    # Compute the RBF kernel matrices
    k_xx = kernel_matrix(x, x, gamma) - jnp.diag(jnp.ones(x.shape[0]))
    k_yy = kernel_matrix(y, y, gamma) - jnp.diag(jnp.ones(x.shape[0]))
    k_xy = kernel_matrix(x, y, gamma) 

    return (k_xx + k_yy).mean() - 2 * k_xy.mean()
