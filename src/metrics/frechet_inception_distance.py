import jax.numpy as jnp

def calculate_fid(x, x_pred):

    # Mean and covariances
    mu_real, std_real = x.mean(axis=0), jnp.cov(x, rowvar=False)
    mu_gen, std_gen = x_pred.mean(axis=0), jnp.cov(x_pred, rowvar=False)

    # Calculate sum of squared differences
    ssd = jnp.sum((mu_real - mu_gen) ** 2)

    # Calculate square root of product of covariances
    cov_mean = jnp.sqrt(std_real @ std_gen)

    # Check and correct imaginary numbers
    if jnp.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Calculate FID
    fid = ssd + jnp.trace(std_real + std_gen - 2 * cov_mean)

    return fid