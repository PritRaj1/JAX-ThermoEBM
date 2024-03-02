import jax
import jax.numpy as jnp
from functools import partial
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])


def EBM_fcn(z, EBM_params, EBM_fwd):

    # Forward pass through the energy-based model
    f_z = EBM_fwd(EBM_params, z)

    return f_z.mean()  # Mean across channels


def prior_grad_log(z, EBM_params, EBM_fwd):
    """
    Function to compute the gradient of the log prior: 
    log[p_a(x)] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - EBM_params: energy-based model parameters
    - EBM_fwd: energy-based model forward pass, --immutable

    Returns:
    - ∇_z( log[p_a(x)] )
    """

    # Find the gradient of the f_a(z) w.r.t. each z
    grad_f = jax.grad(EBM_fcn, argnums=0)(z, EBM_params, EBM_fwd)

    return grad_f - (z / (p0_sig**2))


def log_llood_fcn(z, x, t, GEN_params, GEN_fwd):

    # Forward pass through the generator
    g_z = GEN_fwd(GEN_params, z)

    # Mean squared difference between x and g(z)
    mse = jnp.mean((x - g_z) ** 2)

    # Compute -log[ p_β(x | z)^t ] ∝ -t * [ (x - g(z))^2 / (2 * σ^2) ]
    log_lkhood = -t * (mse) / (2 * pl_sig**2)

    return log_lkhood


def posterior_grad_log(z, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Function to compute the gradient of the log posterior: 
    log[ p(x | z)^t * p(z) ] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - x: batch of data samples
    - t: current temperature
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable

    Returns:
    - ∇_z( log[p_θ(z | x)] ) ∝ ∇_z( log[p(x | z)^t * p(z)] )
    """

    # Find the gradient of the log likelihood w.r.t. each z
    grad_log_llood = jax.grad(log_llood_fcn, argnums=0)(z, x, t, GEN_params, GEN_fwd)

    grad_prior = prior_grad_log(z, EBM_params, EBM_fwd)

    return grad_log_llood + grad_prior
