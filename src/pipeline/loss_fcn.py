import jax
import jax.numpy as jnp
from functools import partial

from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior


def ebm_loss(z_prior, z_posterior, EBM_fwd, EBM_params):
    """
    Function to compute energy-difference loss for the EBM model.
    """

    # Compute the energy of the posterior sample
    en_pos = EBM_fwd(EBM_params, z_posterior)

    # Compute the energy of the prior sample
    en_neg = EBM_fwd(EBM_params, z_prior)

    # Return the difference in energies
    return en_pos - en_neg


def gen_loss(key, x, z, GEN_fwd, GEN_params, pl_sig):
    """
    Function to compute MSE loss for the GEN model.
    """

    # Compute -log[ p_β(x | z) ]; max likelihood training
    key, subkey = jax.random.split(key)
    x_pred = GEN_fwd(GEN_params, z) + (pl_sig * jax.random.normal(subkey, x.shape))
    log_lkhood = (jax.linalg.norm(x - x_pred, axis=-1) ** 2) / (2.0 * pl_sig**2)

    return key, log_lkhood


def TI_EBM_loss_fcn(
    key,
    x,
    EBM_fwd,
    EBM_params,
    GEN_fwd,
    GEN_params,
    pl_sig,
    p0_sig,
    step_size,
    num_steps,
    batch_size,
    num_z,
    temp_schedule,
):
    """
    Function to compute the energy-based model loss using Thermodynamic Integration.

    Please see "discretised thermodynamic integration" using trapezoid rule
    in https://doi.org/10.1016/j.csda.2009.07.025 for details.

    Args:
    - key: PRNG key
    - x: batch of x samples
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - GEN_fwd: generator forward pass, --immutable
    - GEN_params: generator parameters
    - pl_sig: likelihood sigma, --immutable
    - p0_sig: prior sigma, --immutable
    - step_size: step size, --immutable
    - num_steps: number of steps, --immutable
    - batch_size: batch size, --immutable
    - num_z: number of latent space variables, --immutable
    - temp_schedule: temperature schedule, --immutable

    Returns:
    - total_loss: the total loss for the entire thermodynamic integration loop, log(p_a(z))
    """
    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = (0,) + temp_schedule

    total_loss = 0

    # Generate z_posterior for all temperatures
    key, z_posterior = sample_posterior(
        key,
        x,
        EBM_fwd,
        EBM_params,
        GEN_fwd,
        GEN_params,
        pl_sig,
        p0_sig,
        step_size,
        num_steps,
        batch_size,
        num_z,
        temp_schedule,
    )

    for i in range(1, len(temp_schedule) + 1):
        key, z_prior = sample_prior(
            key, EBM_fwd, EBM_params, p0_sig, step_size, num_steps, batch_size, num_z
        )

        z_posterior_t = z_posterior[i - 1]

        loss_current = ebm_loss(z_prior, z_posterior_t, EBM_fwd, EBM_params)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (loss_current + total_loss) * delta_T

    return total_loss, key


def TI_GEN_loss_fcn(
    key,
    x,
    EBM_fwd,
    EBM_params,
    GEN_fwd,
    GEN_params,
    pl_sig,
    p0_sig,
    step_size,
    num_steps,
    batch_size,
    num_z,
    temp_schedule,
):
    """
    Function to compute the generator loss using Thermodynamic Integration.

    Args:
    - key: PRNG key
    - x: batch of x samples
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - GEN_fwd: generator forward pass, --immutable
    - GEN_params: generator parameters
    - pl_sig: likelihood sigma, --immutable
    - p0_sig: prior sigma, --immutable
    - step_size: step size, --immutable
    - num_steps: number of steps, --immutable
    - batch_size: batch size, --immutable
    - num_z: number of latent space variables, --immutable
    - temp_schedule: temperature schedule, --immutable

    Returns:
    - total_loss: the total loss for the entire thermodynamic integration loop, log(p_β(x | z))
    """

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = (0,) + temp_schedule

    total_loss = 0

    # Generate z_posterior for all temperatures
    key, z_posterior = sample_posterior(
        key,
        x,
        EBM_fwd,
        EBM_params,
        GEN_fwd,
        GEN_params,
        pl_sig,
        p0_sig,
        step_size,
        num_steps,
        batch_size,
        num_z,
        temp_schedule,
    )

    for i in range(1, len(temp_schedule) + 1):

        # MSE between g(z) and x, where z ~ p_θ(z|x, t)
        key, loss_current = gen_loss(
            key, x, z_posterior[i - 1], GEN_fwd, GEN_params, pl_sig
        )

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss += 0.5 * (loss_current + total_loss) * delta_T

    return total_loss, key
