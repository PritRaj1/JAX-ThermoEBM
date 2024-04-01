import jax
import jax.numpy as jnp
from jax.numpy.linalg import det, inv
import configparser
from functools import partial

from src.pipeline.loss_computation.loss_helper_fcns import batch_llhood
from src.MCMC_Samplers.grad_log_probs import log_prior_fcn

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])


def KL_div(z1, z2, eps=1e-8, ridge=1e-5):
    """Analytic solution for KL Divergence between power posteriors, assuming multivariate Gaussians."""
    m1 = jnp.mean(z1, axis=0)
    m2 = jnp.mean(z2, axis=0)
    var1 = jnp.cov(z1, rowvar=False) + ridge * jnp.eye(z_channels)
    var2 = jnp.cov(z2, rowvar=False) + ridge * jnp.eye(z_channels)

    KL_div = 0.5 * (
        jnp.log((det(var2) + eps) / (det(var1) + eps))
        + jnp.trace(inv(var2) @ var1)
        + (m1 - m2).T @ inv(var2) @ (m1 - m2)
        - z_channels
    )

    return KL_div


def analytic_KL_bias(
    z_prev, z_curr, *args
):
    """Returns the KL divergence bias term between two adjacent temperatures."""

    z_prev = z_prev.squeeze()
    z_curr = z_curr.squeeze()
    return KL_div(z_prev, z_curr) - KL_div(z_curr, z_prev)


batch_prior = jax.vmap(log_prior_fcn, in_axes=(0, None, None))


def inferred_KL_bias(
    z_prev, z_curr, t_prev, t_curr, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd
):
    """Returns the KL divergence bias term between two adjacent temperatures."""
    posterior_logpdf = lambda z, t: batch_llhood(
        z, x, t, GEN_params, GEN_fwd
    ) + batch_prior(z, EBM_params, EBM_fwd)

    # Sample z' ~ p(z|t_{i-1}) and find the resulting log(p(x|z',t) evaluations
    # E_{p(z|t_{i-1})}[ log(p(x|z,t_{i-1})) - log(p(x|z,t_i)) ]
    KL_high = (
        posterior_logpdf(z_prev, t_prev) - posterior_logpdf(z_prev, t_curr)
    ).mean()

    # Sample z'' ~ p(z|t_i) and find the resulting log(p(x|z'',t) evaluations
    # E_{p(z|t_i)}[ log(p(x|z,t_i)) - log(p(x|z,t_{i-1})) ]
    KL_low = (
        posterior_logpdf(z_curr, t_curr) - posterior_logpdf(z_curr, t_prev)
    ).mean()

    return KL_high - KL_low
