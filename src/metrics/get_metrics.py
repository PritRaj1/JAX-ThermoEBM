import jax.numpy as jnp

def metrcs(x, x_pred):
    """Manual calculation of FID score without torch."""
    # NHWC -> NCHW
    x = jnp.transpose(x, (0, 3, 2, 1))
    x_pred = jnp.transpose(x_pred, (0, 3, 2, 1))

    # Convert for [-1, 1] to [0, 1], image probailities
    x_metric = ((x + 1) / 2).reshape(-1, x.shape[1], x.shape[2], x.shape[3])
    gen_metric = (x_pred + 1) / 2