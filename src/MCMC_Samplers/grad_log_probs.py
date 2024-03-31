import jax
import jax.numpy as jnp
from functools import partial
import optax
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])


### Functions to differentiate ###
def log_prior_fcn(z, EBM_params, EBM_fwd):
    """Compute log(p(z)) ∝ f(z) - 0.5 * (z^2) / (σ^2)"""

    f_z = EBM_fwd(EBM_params, z) - 0.5 * (z**2) / (p0_sig**2)
    return f_z.sum() # log(p(z)) = log(p(z1)) + log(p(z2)) + ... + log(p(zN))

def log_llood_fcn(z, x, t, GEN_params, GEN_fwd):
    """Compute log[ p_β(x | z)^t ] ∝ t * [ - (x - g(z))^2 / (2 * σ^2) ]"""

    g_z = GEN_fwd(GEN_params, z)
    sqr_err = (x - g_z)**2
    log_lkhood = - t * (sqr_err) / (2 * pl_sig**2)
    return log_lkhood.sum() # log(p(x | z)) = log(p(x1 | z)) + log(p(x2 | z)) + ... + log(p(xN | z))

### Grad log probs ###
def prior_grad_log(z, EBM_params, EBM_fwd):
    """
    Compute the gradient of the log prior:
    log[p_α(x)] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - EBM_params: energy-based model parameters
    - EBM_fwd: energy-based model forward pass, --immutable

    Returns:
    - ∇_z( log[p_α(x)] )
    """

    return jax.grad(log_prior_fcn, argnums=0)(z, EBM_params, EBM_fwd)


def posterior_grad_log(z, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Compute the gradient of the log posterior:
    log[ p_β(x | z)^t * p_α(z) ] w.r.t. z.

    Args:
    - z: latent space variable sampled from p0
    - x: batch of data samples
    - t: current temperature
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable

    Returns:
    - ∇_z( log[p_θ(z | x)] ) ∝ ∇_z( log[p_β(x | z)^t * p_α(z)] )
    """

    grad_log_llood = jax.grad(log_llood_fcn, argnums=0)(z, x, t, GEN_params, GEN_fwd)
    grad_prior = prior_grad_log(z, EBM_params, EBM_fwd)
    return grad_log_llood + grad_prior
