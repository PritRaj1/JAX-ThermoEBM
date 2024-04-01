import jax
import jax.numpy as jnp
from jax.lax import scan
from functools import partial
import configparser

from src.pipeline.loss_computation.loss_helper_fcns import mean_GENloss
from src.pipeline.loss_computation.thermo_KL import analytic_KL_bias, inferred_KL_bias
from src.MCMC_Samplers.sample_distributions import sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])
include_bias = str(parser["TEMP"]["INCLUDE_BIAS"])

temp_schedule = jnp.linspace(0, 1, num_temps) ** temp_power
batched_posterior = jax.vmap(
    sample_posterior, in_axes=(0, 0, None, None, None, None, None)
)

# Determine which bias term to use
if include_bias == "analytic":
    get_bias = analytic_KL_bias
elif include_bias == "inferred":
    get_bias = inferred_KL_bias
else:
    get_bias = lambda *args: 0.0


def batch_sample_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
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
    key, z_posterior = batch_sample_posterior(
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
    z_init = batch_sample_posterior(
        key, x, 0.0, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )[1]
    initial_state = (key, 0.0, 0.0, z_init, 0)
    (_, _, _, _, _), temp_losses = scan(
        f=scan_loss, init=initial_state, xs=temp_schedule
    )

    return -0.5 * temp_losses.sum()
