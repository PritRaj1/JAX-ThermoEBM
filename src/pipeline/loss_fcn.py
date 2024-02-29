import jax
import jax.numpy as jnp
from functools import partial
import configparser

from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser['SIGMAS']['LKHOOD_SIGMA'])

def ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd):
    """
    Function to compute energy-difference loss for the EBM model.
    """

    # Compute the energy of the posterior sample
    en_pos = EBM_fwd(EBM_params, z_posterior)

    # Compute the energy of the prior sample
    en_neg = EBM_fwd(EBM_params, z_prior)

    # Return the difference in energies
    return jnp.mean(en_pos - en_neg, axis=-1).squeeze()


def gen_loss(key, x, z, GEN_params, GEN_fwd):
    """
    Function to compute MSE loss for the GEN model.
    """

    # Compute -log[ p_β(x | z) ]; max likelihood training
    key, subkey = jax.random.split(key)
    x_pred = GEN_fwd(GEN_params, z) + (pl_sig * jax.random.normal(subkey, x.shape))
    log_lkhood = (jnp.linalg.norm(x - x_pred, axis=(1,2)) ** 2) / (2.0 * pl_sig**2)
    log_lkhood = jnp.mean(log_lkhood, axis=1)

    return key, log_lkhood

@partial(jax.jit, static_argnums=(4,5,6))
def TI_EBM_loss_fcn(
    key,
    x,
    EBM_params,
    GEN_params,
    EBM_fwd,
    GEN_fwd,
    temp_schedule
):
    """
    Function to compute the energy-based model loss using Thermodynamic Integration.

    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    Args:
    - key: PRNG key
    - x: sample of x
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - temp_schedule: temperature schedule, --immutable

    Returns:
    - total_loss: the total loss for the entire thermodynamic integration loop, log(p_a(z))
    """

    total_loss = jnp.array(0)

    # Generate z_posterior for all temperatures
    key, z_posterior = sample_posterior(
        key,
        x,
        EBM_params,
        GEN_params,
        EBM_fwd,
        GEN_fwd,
        temp_schedule
    )

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = (0,) + temp_schedule

    for i in range(1, len(temp_schedule)):
        key, z_prior = sample_prior(
            key, EBM_params, EBM_fwd
        )

        z_posterior_t = z_posterior[i - 1]

        loss_current = ebm_loss(z_prior, z_posterior_t, EBM_params, EBM_fwd)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (loss_current + total_loss) * delta_T

    return total_loss, key

@partial(jax.jit, static_argnums=(4,5,6))
def TI_GEN_loss_fcn(
    key,
    x,
    EBM_params,
    GEN_params,
    EBM_fwd,
    GEN_fwd,
    temp_schedule
):
    """
    Function to compute the generator loss using Thermodynamic Integration.

    Args:
    - key: PRNG key
    - x: batch of x samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - temp_schedule: temperature schedule, --immutable

    Returns:
    - total_loss: the total loss for the entire thermodynamic integration loop, log(p_β(x | z))
    """

    total_loss = jnp.zeros(x.shape[0])

    # Generate z_posterior for all temperatures
    key, z_posterior = sample_posterior(
        key,
        x,
        EBM_params,
        GEN_params,
        EBM_fwd,
        GEN_fwd,
        temp_schedule
    )

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = (0,) + temp_schedule

    for i in range(1, len(temp_schedule)):

        z_posterior_t = z_posterior[i - 1]

        # MSE between g(z) and x, where z ~ p_θ(z|x, t)
        key, loss_current = gen_loss(
            key, x, z_posterior_t, GEN_params, GEN_fwd
        )

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (loss_current + total_loss) * delta_T

    return total_loss, key
