import jax
import jax.numpy as jnp
from jax.lax import scan
from functools import partial
import configparser

from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])


def ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd):
    """
    Function to compute energy-difference loss for the EBM model.
    """

    # Compute the energy of the posterior sample
    en_pos = EBM_fwd(EBM_params, z_posterior)

    # Compute the energy of the prior sample
    en_neg = EBM_fwd(EBM_params, z_prior)

    difference = en_pos - en_neg

    return difference.sum()


def gen_loss(key, x, z, GEN_params, GEN_fwd):
    """
    Function to compute MSE loss for the GEN model.
    """

    key, subkey = jax.random.split(key)

    # Generate a sample from the generator
    x_pred = GEN_fwd(GEN_params, z) + (pl_sig * jax.random.normal(subkey, x.shape))

    # Sum of element-wise squared differences (l2 norm squared)
    sqr_err = jnp.sum((x - x_pred) ** 2)

    # Compute -log[ p_β(x | z) ] = 1/2 * (x - g(z))^2 / σ^2
    log_lkhood = sqr_err / (2.0 * pl_sig**2)

    return key, log_lkhood.sum()


def ThermoEBM_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):
    """
    Function to compute the themodynamic integration loss for the EBM model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:
    ∫[a, b] f(x) dx ≈ 1/2 * (f(a) + f(b)) * ∇T
    We accumulate this in a scan loop over the temperature schedule.
    This is then summed at the end to compute the integral over the temperature schedule.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - temp_schedule: temperature schedule

    Returns:
    - total_loss: the total EBM loss for the entire thermodynamic integration loop, log(p_a(z))
    """

    def loss(carry, t):

        # Extract the key, temperature, and loss carried from the previous temperature
        key_i, t_prev, loss_prev = carry

        # Sample from the prior distribution
        key_i, z_prior = sample_prior(key_i, EBM_params, EBM_fwd)

        # Sample from the posterior distribution tempered by the current temperature
        key_i, z_posterior = sample_posterior(
            key_i, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
        )

        # Compute the current loss
        current_loss = ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd)

        # ∇T = t_i - t_{i-1}
        delta_T = t - t_prev

        # Accumulate 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        temperature_loss = 0.5 * (current_loss + loss_prev) * delta_T

        return (key_i, t, current_loss), temperature_loss

    initial_state = (key, 0, 0)

    (_, _, _), temp_losses = scan(f=loss, init=initial_state, xs=temp_schedule)

    return temp_losses.sum()


def ThermoGEN_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):
    """
    Function to compute the themodynamic integration loss for the GEN model.
    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    To integrate over temperatures, we use the trapezoid rule:
    ∫[a, b] f(x) dx ≈ 1/2 * (f(a) + f(b)) * ∇T
    We accumulate this in a scan loop over the temperature schedule.
    This is then summed at the end to compute the integral over the temperature schedule.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - temp_schedule: temperature schedule

    Returns:
    - total_loss: the total GEN loss for the entire thermodynamic integration loop, log(p_β(x|z))
    """

    def loss(carry, t):

        # Extract the key, temperature, and loss carried from the previous temperature
        key_i, t_prev, loss_prev = carry

        # Sample from the posterior distribution, tempered by the current temperature
        key_i, z_posterior = sample_posterior(
            key_i, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
        )

        # Compute the loss
        key_i, current_loss = gen_loss(key_i, x, z_posterior, GEN_params, GEN_fwd)

        # ∇T = t_i - t_{i-1}
        delta_T = t - t_prev

        # log[ p_β(x|z,t) ] = 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        temperature_loss = 0.5 * (current_loss + loss_prev) * delta_T

        return (key_i, t, current_loss), temperature_loss

    initial_state = (key, 0, 0)

    (_, _, _), temp_losses = scan(f=loss, init=initial_state, xs=temp_schedule)

    return temp_losses.sum()
