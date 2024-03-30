import jax
import jax.numpy as jnp
from jax.lax import scan
from jax.numpy.linalg import det, inv
from functools import partial
import configparser

from src.pipeline.loss_computation.loss_helper_fcns import mean_GENloss, llhood
from src.MCMC_Samplers.sample_distributions import sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])
include_bias = bool(parser["TEMP"]["INCLUDE_BIAS"])

temp_schedule = jnp.linspace(0, 1, num_temps) ** temp_power
batched_posterior = jax.vmap(
    sample_posterior, in_axes=(0, 0, None, None, None, None, None)
)


# def KL_div(z1, z2, eps=1e-8, ridge=1e-5):
#     """Analytic solution for KL Divergence between power posteriors, assuming multivariate Gaussians."""
#     m1 = jnp.mean(z1, axis=0)
#     m2 = jnp.mean(z2, axis=0)
#     var1 = jnp.cov(z1, rowvar=False) + ridge * jnp.eye(z_channels)
#     var2 = jnp.cov(z2, rowvar=False) + ridge * jnp.eye(z_channels)

#     KL_div = 0.5 * (
#         jnp.log((det(var2) + eps) / (det(var1) + eps))
#         + jnp.trace(inv(var2) @ var1)
#         + (m1 - m2).T @ inv(var2) @ (m1 - m2)
#         - z_channels
#     )


#     return KL_div


# def get_KL_bias(z_prev, z_curr):
#     """Returns the KL divergence bias term between two adjacent temperatures."""

#     return KL_div(z_prev, z_curr) - KL_div(z_curr, z_prev)


def get_llhood(x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    return t * llhood(x, z, GEN_params, GEN_fwd) + EBM_fwd(EBM_params, z).sum()


batch_llhood = jax.vmap(get_llhood, in_axes=(0, 0, None, None, None, None, None))


def get_KL_bias(
    z_prev, z_curr, t_prev, t_curr, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd
):

    def logpdf(z, t):
        value = batch_llhood(x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
        return value - jax.scipy.special.logsumexp(value, axis=0)

    # Sample z' ~ p(z|t_{i-1}) and find the resulting log(p(x|z',t) evaluations
    # E_{p(z|t_{i-1})}[ log(p(x|z,t_{i-1})) - log(p(x|z,t_i)) ]
    KL_high = (logpdf(z_prev, t_prev) - logpdf(z_prev, t_curr)).mean()

    # Sample z'' ~ p(z|t_i) and find the resulting log(p(x|z'',t) evaluations
    # E_{p(z|t_i)}[ log(p(x|z,t_i)) - log(p(x|z,t_{i-1})) ]
    KL_low = (logpdf(z_curr, t_curr) - logpdf(z_curr, t_prev)).mean()

    return KL_high - KL_low


if include_bias:
    get_bias = get_KL_bias
else:
    get_bias = lambda x, y: 0.0


def get_batched_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns a batch of samples from the posterior distribution at a given temperature."""

    # Sample batch amount of z_posterior samples
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    z_posterior = batched_posterior(
        subkey_batch, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    return key, z_posterior


def thermo_scan_loop(carry, t, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Scan step to compute the expected llhood loss at one temperature."""

    # Parse the carry state
    key, t_prev, prev_loss, prev_z, keep_KL = carry

    # Get liklihood, E_{z|x,t}[ log(p_β(x|z,t)) ]
    key, z_posterior = get_batched_posterior(
        key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )
    current_loss = mean_GENloss(x, z_posterior, GEN_params, GEN_fwd)

    # ∇T = t_i - t_{i-1}
    delta_T = t - t_prev

    # ((L(t_i) + L(t_{i-1})) * ∇T) + (KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}])
    temperature_loss = (current_loss + prev_loss) * delta_T + get_bias(
        prev_z, z_posterior, t_prev, t, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    ) * keep_KL  # Do not include KL divergence in first iter, (area is 0 between t=0 and t=0)

    # Push tempered loss to the stack and carry over the current state
    return (key, t, current_loss, z_posterior, 1), temperature_loss


def thermo_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Function to compute the themodynamic integration loss for the GEN model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:

    log p_θ(x) =
            1/2 * Σ [ ΔT (E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] )
            + 1/2 * Σ [ KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}] ] ]

    The terms within the summations are accumulated in a scan loop over the temperature schedule.
    This is then summed at the end to compute the integral over the temperature schedule.
    The KL Divergence term is included to account for bias, but is not strictly necessary.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable

    Returns:
    - total_loss: the total lkhood loss for the entire thermodynamic integration loop, -∫ E_{z|x,t}[ log(p_β(x|z)) ] dt
    """

    # Wrap the themodynamic loop in a partial function to exploit partial immutability
    scan_loss = partial(
        thermo_scan_loop,
        x=x,
        EBM_params=EBM_params,
        GEN_params=GEN_params,
        EBM_fwd=EBM_fwd,
        GEN_fwd=GEN_fwd,
    )

    # Scan along each temperature and stack the losses
    z_init = get_batched_posterior(
        key, x, 0.0, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )[1]
    initial_state = (key, 0.0, 0.0, z_init, 0)
    (_, _, _, _, _), temp_losses = scan(
        f=scan_loss, init=initial_state, xs=temp_schedule
    )

    return -0.5 * temp_losses.sum()
