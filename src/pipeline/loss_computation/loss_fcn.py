from jax.lax import scan
import jax.numpy as jnp
from functools import partial
import configparser
from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior
from src.pipeline.loss_computation.pure_loss import ebm_loss, gen_loss
from optax import kl_divergence as kl_div
from jax.scipy.stats import gaussian_kde as kde

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])

if temp_power > 0:
    print("Using Temperature Schedule with Power: {}".format(temp_power))
    temp_schedule = jnp.linspace(0, 1, num_temps) ** temp_power
    print("Temperature Schedule: {}".format(temp_schedule))
else:
    print("Using no Thermodynamic Integration, defaulting to Vanilla Model")


def vanilla_EBM_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """-(E_{z|x}[ f(z) ] - E_{z}[ f(z) ])"""

    # Sample from the prior distribution, do not replace the key to make sure llhood loss uses same posterior
    _, z_prior = sample_prior(key, EBM_params, EBM_fwd)

    # Sample from the untempered posterior distribution
    key, z_posterior = sample_posterior(
        key, x, 1, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )   

    # Negate to maximise the likelihood
    return -ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd)


def vanilla_GEN_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns -log[ p_β(x | z, t=1) ]"""

    # Sample from the untempered posterior distribution
    key, z_posterior = sample_posterior(
        key, x, 1, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )
    
    # Negate to maximise the likelihood
    return -gen_loss(key, x, z_posterior, GEN_params, GEN_fwd)[1]


def thermo_scan_loop(carry, t, x, EBM_params, EBM_fwd, GEN_params, GEN_fwd):
    """Loop step to compute the GEN loss at one temperature."""

    key_i, t_prev, prev_loss, prev_z = carry

    # Sample from the posterior distribution tempered by the current temperature
    key_i, z_posterior = sample_posterior(
        key_i, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    # Current loss = E_{z|x,t}[ log[ p_β(x | z) ] ]
    key_i, current_loss = gen_loss(key_i, x, z_posterior, GEN_params, GEN_fwd)

    # ∇T = t_i - t_{i-1}
    delta_T = t - t_prev

    # Compute the KDEs for the current and previous samples for KL divergence calculation
    z_posterior = z_posterior.squeeze()
    prev_kde = kde(prev_z)
    current_kde = kde(z_posterior)

    # ((L(t_i) + L(t_{i-1})) * ∇T) + (KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}])
    temperature_loss = (current_loss + prev_loss) * delta_T + (
        kl_div(prev_kde.logpdf(prev_z), current_kde.pdf(z_posterior))
        - kl_div(current_kde.logpdf(z_posterior), prev_kde.pdf(prev_z))
    )

    # Push temp loss to the stack and carry over the current state
    return (key_i, t, current_loss, z_posterior), temperature_loss


def Thermo_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Function to compute the themodynamic integration loss for the GEN model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:

    log p_θ(x) =
            1/2 * Σ [ ΔT [E_{z|x,t_i}[ log p_β(x | z) ] + E_{z|x,t_{i-1}}[ log p_β(x | z) ] ]
            + 1/2 * Σ [ KL[z_{i-1} || z_i] - KL[z_i || z_{i-1}] ] ]

    The term within the summations are accumulated in a scan loop over the temperature schedule.
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
        EBM_fwd=EBM_fwd,
        GEN_params=GEN_params,
        GEN_fwd=GEN_fwd,
    )

    # Initialise t = 0 state
    key, z_init = sample_posterior(key, x, 0, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
    loss_init = gen_loss(key, x, z_init, GEN_params, GEN_fwd)[1]
    initial_state = (key, 0, loss_init, z_init.squeeze())

    # Scan over the temperature schedule to compute the loss at each temperature
    (_, _, _, _), temp_losses = scan(f=scan_loss, init=initial_state, xs=temp_schedule)

    # Sum the stacked losses over all temperature intervals and negate to get integrated llhood loss
    return - 0.5 * temp_losses.sum()
