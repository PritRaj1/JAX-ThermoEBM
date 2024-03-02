import jax
import jax.numpy as jnp
from functools import partial
import configparser
from jax.lax import scan

from src.MCMC_Samplers.grad_log_probs import prior_grad_log, posterior_grad_log

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])

prior_steps = int(parser["MCMC"]["E_SAMPLE_STEPS"])
prior_s = float(parser["MCMC"]["E_STEP_SIZE"])
posterior_steps = int(parser["MCMC"]["G_SAMPLE_STEPS"])
posterior_s = float(parser["MCMC"]["G_STEP_SIZE"])


def update_step(key, x, grad_f, s):
    """Update the current state of the sampler."""
    x += s * s * grad_f

    key, subkey = jax.random.split(key)
    x += jnp.sqrt(2) * s * jax.random.normal(subkey, x.shape)

    return key, x


def sample_p0(key):
    """Sample from the prior distribution."""

    key, subkey = jax.random.split(key)
    return key, p0_sig * jax.random.normal(subkey, (1, 1, z_channels))


def sample_prior(key, EBM_params, EBM_fwd):
    """
    Sample from the prior distribution.

    Args:
    - key: PRNG key
    - EBM_params: energy-based model parameters
    - EBM_fwd: energy-based model forward pass, --immutable

    Returns:
    - key: PRNG key
    - z: latent space variable sampled from p_a(x)
    """
    
    def MCMC_steps(carry, _):

        jax.debug.breakpoint()

        key, z = carry
        grad_f = prior_grad_log(z, EBM_params, EBM_fwd)
        key, z = update_step(key, z, grad_f, prior_s)
        return (key, z), None

    key0, z0 = sample_p0(key)
    (final_key, final_z), _ = scan(MCMC_steps, (key0, z0), None, length=prior_steps)

    return final_key, final_z


def sample_posterior(
    key,
    x,
    t,
    EBM_params,
    GEN_params,
    EBM_fwd,
    GEN_fwd
):
    """
    Sample from the posterior distribution.

    Args:
    - key: PRNG key
    - x: batch of data samples
    - t: current temperature
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable
    - posterior_steps: number of MCMC steps
    - posterior_grad_log: function for computing gradient of log posterior
    - update_step: function for updating MCMC state
    - sample_p0: function for sampling initial state
    - posterior_s: MCMC step size

    Returns:
    - key: PRNG key
    - z_samples: samples from the posterior distribution indexed by temperature
    """

   
    def MCMC_steps(carry, _):
        
        key, z = carry
        grad_f = posterior_grad_log(z, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
        key, z = update_step(key, z, grad_f, posterior_s)
        return (key, z), None

    key0, z0 = sample_p0(key)

    (final_key, final_z), _ = scan(MCMC_steps, (key0, z0), None, length=posterior_steps)
    
    return final_key, final_z
