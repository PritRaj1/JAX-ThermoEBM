import jax
import jax.numpy as jnp
from functools import partial


def prior_grad_log(z, EBM_fwd, EBM_params, p0_sig):
    """
    Function to compute the gradient of the log prior: log[p_a(x)] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - p0_sig: prior sigma, --immutable

    Returns:
    - ∇_z( log[p_a(x)] )
    """
    # Find the gradient of the f_a(z) w.r.t. z
    grad_f = jax.grad(EBM_fwd)(EBM_params, z)[1]

    return grad_f - (z / (p0_sig**2))


def posterior_grad_log(
    z, x, t, EBM_fwd, EBM_params, GEN_fwd, GEN_params, pl_sig, p0_sig
):
    """
    Function to compute the gradient of the log posterior: log[ p(x | z)^t * p(z) ] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - x: batch of data samples
    - t: current temperature
    EBM_fwd: energy-based model forward pass, --immutable
    EBM_params: energy-based model parameters
    GEN_fwd: generator forward pass, --immutable
    GEN_params: generator parameters
    pl_sig: likelihood sigma, --immutable
    p0_sig: prior sigma, --immutable

    Returns:
    - ∇_z( log[p_θ(z | x)] ) ∝ ∇_z( log[p(x | z)^t * p(z)] )
    """

    def log_llood_fcn(z, x):
        g_z = GEN_fwd(GEN_params, z)
        return -t * (jnp.linalg.norm(x - g_z, axis=-1) ** 2) / (2.0 * pl_sig**2)

    grad_log_llood = jax.grad(log_llood_fcn)(z, x)[0]

    grad_prior = prior_grad_log(z, EBM_fwd, EBM_params, p0_sig)

    return grad_log_llood + grad_prior
