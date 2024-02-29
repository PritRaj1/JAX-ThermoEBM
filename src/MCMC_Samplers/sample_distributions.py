import jax
import jax.numpy as jnp
from functools import partial
from src.MCMC_Samplers.grad_log_probs import prior_grad_log, posterior_grad_log


def update_step(key, x, grad_f, s):
    """Update the current state of the sampler."""
    x += s * grad_f

    key, subkey = jax.random.split(key)
    x += jnp.sqrt(2 * s) * jax.random.normal(subkey, x.shape)

    return key, x


def sample_p0(key, p0_sig, batch_size, z_channels):
    """Sample from the prior distribution."""

    key, subkey = jax.random.split(key)
    return key, p0_sig * jax.random.normal(subkey, (batch_size, 1, 1, z_channels)) 


def sample_prior(
    key, EBM_fwd, EBM_params, p0_sig, step_size, num_steps, batch_size, z_channels
):
    """
    Sample from the prior distribution.

    Args:
    - key: PRNG key
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - p0_sig: prior sigma, --immutable
    - step_size: step size, --immutable
    - num_steps: number of steps, --immutable
    - batch_size: batch size, --immutable

    Returns:
    - key: PRNG key
    - z: latent space variable sampled from p_a(x)
    """

    key, z = sample_p0(key, p0_sig, batch_size, z_channels)

    for k in range(num_steps):
        grad_f = prior_grad_log(z, EBM_fwd, EBM_params, p0_sig)
        key, z = update_step(key, z, grad_f, step_size)

    return key, z


def sample_posterior(
    key,
    data,
    EBM_fwd,
    EBM_params,
    GEN_fwd,
    GEN_params,
    pl_sig,
    p0_sig,
    step_size,
    num_steps,
    batch_size,
    z_channels,
    temp_schedule,
):
    """
    Sample from the posterior distribution.

    Args:
    - key: PRNG key
    - data: batch of data samples
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - GEN_fwd: generator forward pass, --immutable
    - GEN_params: generator parameters
    - pl_sig: likelihood sigma, --immutable
    - p0_sig: prior sigma, --immutable
    - step_size: step size, --immutable
    - num_steps: number of steps, --immutable
    - batch_size: batch size, --immutable
    - temp_schedule: temperature schedule, --immutable

    Returns:
    - key: PRNG key
    - z_samples: samples from the posterior distribution indexed by temperature
    """

    z_samples = jnp.zeros((len(temp_schedule), batch_size, 1, 1, z_channels))

    for idx, t in enumerate(temp_schedule):
        key, z = sample_p0(key, p0_sig, batch_size, z_channels)

        for k in range(num_steps):
            grad_f = posterior_grad_log(
                z, data, t, EBM_fwd, EBM_params, GEN_fwd, GEN_params, pl_sig, p0_sig
            )
            key, z = update_step(key, z, grad_f, step_size)

        z_samples = z_samples.at[idx].set(z)

    return key, z_samples
