import jax.numpy as jnp

def kernel(x, y, sigma=1):
    """RBF kernel."""
    return jnp.exp(-jnp.sum((x - y) ** 2) / (2 * sigma ** 2))

def calculate_mmd(x, x_pred):
    """Emprical maximum mean discrepancy."""
    n, m = x.shape[0], x_pred.shape[0]

    # Calculate the kernel
    k_xx = jnp.mean(kernel(x[i], x[j]) for i in range(n) for j in range(n))
    k_yy = jnp.mean(kernel(x_pred[i], x_pred[j]) for i in range(m) for j in range(m))
    k_xy = jnp.mean(kernel(x[i], x_pred[j]) for i in range(n) for j in range(m))

    # Calculate the MMD
    mmd = k_xx + k_yy - 2 * k_xy

    return mmd