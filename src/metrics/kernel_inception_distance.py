import jax.numpy as jnp
from src.metrics.inception_network import InceptionV3

def kernel(x, y, sigma=0.1):
    """RKHS kernel."""
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * sigma ** 2))

def calculate_kid(real_features, fake_features):
    """
    Emprical maximum mean discrepancy.
    
     	
    https://doi.org/10.48550/arXiv.2206.10935
    """
    n, m = real_features.shape[0], fake_features.shape[0]

    # Calculate the kernel
    k_xx = jnp.mean(kernel(real_features[i], fake_features[j]) for i in range(n) for j in range(n))
    k_yy = jnp.mean(kernel(fake_features[i], fake_features[j]) for i in range(m) for j in range(m))
    k_xy = jnp.mean(kernel(real_features[i], fake_features[j]) for i in range(n) for j in range(m))

    # Calculate the MMD
    mmd = k_xx + k_yy - 2 * k_xy

    return mmd