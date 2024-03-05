import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import configparser
import optax

from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])


def ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd):
    """Function to compute energy-difference loss for the EBM model."""

    # Compute the energy of the posterior sample
    en_pos = EBM_fwd(EBM_params, stop_gradient(z_posterior)).mean()

    # Compute the energy of the prior sample
    en_neg = EBM_fwd(EBM_params, stop_gradient(z_prior)).mean()

    return (en_pos - en_neg)


def gen_loss(key, x, z, GEN_params, GEN_fwd):
    """Function to compute MSE loss for the GEN model."""

    # Generate a sample from the generator
    key, subkey = jax.random.split(key)
    x_pred = GEN_fwd(GEN_params, stop_gradient(z)) + (
        pl_sig * jax.random.normal(subkey, x.shape)
    )

    # Compute -log[ p_β(x | z) ] = 1/2 * (x - g(z))^2 / σ^2
    mse = jnp.mean(optax.l2_loss(x, x_pred))
    log_lkhood = mse / (2.0 * pl_sig**2)

    return key, log_lkhood


def EBM_loop(carry, t, x, EBM_params, EBM_fwd, GEN_params, GEN_fwd):
    """Loop step to compute the EBM loss at one temperature."""

    key_i, t_prev, prev_loss = carry

    # Sample from the prior distribution, do not replace the key
    _, z_prior = sample_prior(key_i, EBM_params, EBM_fwd)

    # Sample from the posterior distribution tempered by the current temperature
    key_i, z_posterior = sample_posterior(
        key_i, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    # Current loss = E_{z|x}[ f(z) ] - E_{z}[ f(z) ]
    current_loss = ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd)

    # ∇T = t_i - t_{i-1}
    delta_T = t - t_prev

    # 1/2 * (L(x_i) + L(x_{i-1})) * ∇T
    temperatue_loss = 0.5 * (current_loss + prev_loss) * delta_T

    # Add temp loss to the stack and carry over the current state
    return (key_i, t, current_loss), temperatue_loss


def GEN_loop(carry, t, x, EBM_params, EBM_fwd, GEN_params, GEN_fwd):
    """Loop step to compute the GEN loss at one temperature."""

    key_i, t_prev, prev_loss = carry

    # Sample from the posterior distribution tempered by the current temperature
    key_i, z_posterior = sample_posterior(
        key_i, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    # Current loss = -log[ p_β(x | z) ]
    key_i, current_loss = gen_loss(key_i, x, z_posterior, GEN_params, GEN_fwd)

    # ∇T = t_i - t_{i-1}
    delta_T = t - t_prev

    # # 1/2 * (L(x_i) + L(x_{i-1})) * ∇T
    temperature_loss = 0.5 * (current_loss + prev_loss) * delta_T

    # Add temp loss to the stack and carry over the current state
    return (key_i, t, current_loss), temperature_loss
