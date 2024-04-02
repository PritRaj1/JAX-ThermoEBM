import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
from jax.scipy.linalg import det, inv
import configparser

from src.loss_computation.loss_helper_fcns import batched_marginal_llhood

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


def analytic_KL_bias(key, z_prev, z_curr, *args):
    """Returns the KL divergence bias term between two adjacent temperatures."""

    z_prev = z_prev.squeeze()
    z_curr = z_curr.squeeze()
    return KL_div(z_prev, z_curr) - KL_div(z_curr, z_prev)


def inferred_KL_bias(
    key, z_prev, z_curr, t_prev, t_curr, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd
):

    # p(z|x, t) = p(x|z, t) * p(z|t) / p(x|t)
    def expected_posterior(prior_key, z, t):
        llhood = batched_marginal_llhood(prior_key, x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
        llhood -= jax.scipy.special.logsumexp(stop_gradient(llhood))
        return llhood.mean() # E_{p(z|x,t)}[ log(p(z|x,t)) ]

    keys = jax.random.split(key, 4)

    # Sample z' ~ p(z|x,t_{i-1}) and find the resulting log(p(z'|x,t) evaluations
    # E_{p(z'|x, t_{i-1})}[ log(p(z|x, t_{i-1})) - log(p(z|x,t_i)) ]
    KL_high = expected_posterior(prior_key=keys[0], z=z_prev, t=t_prev) - expected_posterior(
        prior_key=keys[1], z=z_prev, t=t_curr
    )

    # Sample z'' ~ p(z|x,t_i) and find the resulting log(p(z''|x,t) evaluations
    # E_{p(z''|t_i)}[ log(p(x|z,t_i)) - log(p(x|z,t_{i-1})) ]
    KL_low = expected_posterior(prior_key=keys[2], z=z_curr, t=t_curr) - expected_posterior(
        prior_key=keys[3], z=z_curr, t=t_prev
    )

    return KL_high - KL_low
