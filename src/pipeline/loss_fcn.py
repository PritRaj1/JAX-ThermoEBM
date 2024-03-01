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
    log_lkhood = jnp.mean(log_lkhood, axis=-1)

    return key, log_lkhood.squeeze()

# @partial(jax.jit, static_argnums=(4,5,6))
def ThermodynamicIntegrationLoss(
        key,
        x,
        EBM_params,
        GEN_params,
        EBM_fwd,
        GEN_fwd,
        temp_schedule
):
    
    total_loss_ebm = 0
    total_loss_gen = 0

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = (0,) + temp_schedule

    for i in range(1, len(temp_schedule)):
        key, z_prior = sample_prior(
            key, EBM_params, EBM_fwd
        )

        key, z_posterior = sample_posterior(
            key, x, temp_schedule[i], EBM_params, GEN_params, EBM_fwd, GEN_fwd
        )

        loss_current_ebm = ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd)
        key, loss_current_gen = gen_loss(key, x, z_posterior, GEN_params, GEN_fwd)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss_ebm += 0.5 * (loss_current_ebm + total_loss_ebm) * delta_T
        total_loss_gen += 0.5 * (loss_current_gen + total_loss_gen) * delta_T

    return jnp.asarray([total_loss_ebm, total_loss_gen])