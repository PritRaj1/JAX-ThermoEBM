import jax
import jax.numpy as jnp
from jax.nn import softmax, log_softmax
from jax.lax import scan
from optax import kl_divergence as kl_div
from functools import partial
import configparser

from src.pipeline.loss_computation.loss_helper_fcns import mean_GENloss
from src.MCMC_Samplers.sample_distributions import sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])
temp_schedule = jnp.linspace(0, 1, num_temps) ** temp_power
batched_posterior = jax.vmap(
    sample_posterior, in_axes=(0, 0, None, None, None, None, None)
)


def get_batched_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Function to sample a batch from the posterior distribution at a given temperature."""

    # Sample batch amount of z_posterior samples
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    z_posterior = batched_posterior(
        subkey_batch, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    return key, z_posterior


def thermo_scan_loop(carry, t, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Loop step to compute the GEN loss at one temperature."""

    # Parse the carry state
    key, t_prev, prev_loss, prev_z = carry

    # Get liklihood, E_{z|x,t}[ log(p_β(x|z,t)) ]
    key, z_posterior = get_batched_posterior(
        key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )
    current_loss = mean_GENloss(x, z_posterior, GEN_params, GEN_fwd)
    z_posterior = z_posterior.squeeze()

    # ∇T = t_i - t_{i-1}
    delta_T = t - t_prev

    # ((L(t_i) + L(t_{i-1})) * ∇T) + (KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}])
    temperature_loss = (current_loss + prev_loss) * delta_T + (
        kl_div(log_softmax(prev_z), softmax(z_posterior)).mean()
        - kl_div(log_softmax(z_posterior), softmax(prev_z)).mean()
    )

    # Push tempered loss to the stack and carry over the current state
    return (key, t, current_loss, z_posterior), temperature_loss


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

    # Initialise z at the first temperature to avoid nan KL divergence
    key, z_init = get_batched_posterior(
        key, x, 0, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )
    initial_state = (key, 0.0, 0.0, z_init.squeeze())

    # Scan along each temperature and stack the losses
    (_, _, _, _), temp_losses = scan(f=scan_loss, init=initial_state, xs=temp_schedule)

    return -0.5 * temp_losses.sum()
