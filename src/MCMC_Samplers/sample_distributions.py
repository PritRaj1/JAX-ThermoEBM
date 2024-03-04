import jax
from functools import partial
import configparser
from jax.lax import scan

from src.MCMC_Samplers.langevin_updates import langevin_prior, langevin_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
z_channels = int(parser["EBM"]["Z_CHANNELS"])

prior_steps = int(parser["MCMC"]["E_SAMPLE_STEPS"])
prior_s = float(parser["MCMC"]["E_STEP_SIZE"])
posterior_steps = int(parser["MCMC"]["G_SAMPLE_STEPS"])
posterior_s = float(parser["MCMC"]["G_STEP_SIZE"])


def sample_p0(key):
    """Sample from the noise prior distribution."""

    key, subkey = jax.random.split(key)
    return key, p0_sig * jax.random.normal(subkey, (1, 1, z_channels))


def get_noise_step(key, num_steps, step_size, shape):
    """Get all noises for the MCMC steps."""

    key, subkey = jax.random.split(key)
    return key, step_size * jax.random.normal(subkey, (num_steps,) + shape)


def sample_prior(key, EBM_params, EBM_fwd):
    """
    Sample from the exponentially-tilted prior distribution.

    Args:
    - key: PRNG key
    - EBM_params: energy-based model parameters
    - EBM_fwd: energy-based model forward pass, --immutable

    Returns:
    - key: PRNG key
    - z: latent space variable sampled from p_a(x)
    """

    # Wrap the langevin update function with constants for scanning
    scan_MCMC = partial(langevin_prior, EBM_params=EBM_params, EBM_fwd=EBM_fwd)

    # Sample all noise at once, to avoid reseeding the PRNG and reduce overhead
    key, z0 = sample_p0(key)
    key, noise = get_noise_step(key, prior_steps, prior_s, z0.shape)

    # Scan along the noise to iteratively update z
    z_prior, _ = scan(scan_MCMC, z0, noise, length=prior_steps)

    return key, z_prior


def sample_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Sample from the posterior distribution.

    Args:
    - key: PRNG key
    - x: data samples
    - t: current temperature
    - EBM_params: energy-based model parameters
    - GEN_params: generator parameters
    - EBM_fwd: energy-based model forward pass, --immutable
    - GEN_fwd: generator forward pass, --immutable

    Returns:
    - key: PRNG key
    - z_samples: samples from the posterior distribution indexed by temperature
    """

    # Wrap the langevin update function with constants for scanning
    scan_MCMC = partial(
        langevin_posterior,
        x=x,
        t=t,
        EBM_params=EBM_params,
        GEN_params=GEN_params,
        EBM_fwd=EBM_fwd,
        GEN_fwd=GEN_fwd,
    )

    # Sample all noise at once, to avoid reseeding the PRNG and reduce overhead
    key, z0 = sample_p0(key)
    key, noise = get_noise_step(key, posterior_steps, posterior_s, z0.shape)

    # Scan along the noise to iteratively update z
    z_posterior, _ = scan(scan_MCMC, z0, noise, length=posterior_steps)

    return key, z_posterior
