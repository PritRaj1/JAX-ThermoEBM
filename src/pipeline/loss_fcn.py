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

    # Return the difference in energies
    return jnp.mean(en_pos - en_neg, axis=-1).sum()


def gen_loss(key, x, z, GEN_params, GEN_fwd):
    """
    Function to compute MSE loss for the GEN model.
    """

    # Compute -log[ p_β(x | z) ]; max likelihood training
    key, subkey = jax.random.split(key)
    x_pred = GEN_fwd(GEN_params, z) + (pl_sig * jax.random.normal(subkey, x.shape))
    log_lkhood = (jnp.linalg.norm(x - x_pred, axis=(1, 2)) ** 2) / (2.0 * pl_sig**2)
    log_lkhood = jnp.mean(log_lkhood, axis=-1)

    return key, log_lkhood.sum()


# def ThermodynamicIntegrationLoss(
#     key0, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule
# ):
#     """
#     Function to compute the energy-based model loss using Thermodynamic Integration.

#     Please see "discretised thermodynamic integration" using trapezoid rule
#     in https://doi.org/10.1016/j.csda.2009.07.025 for details.

#     Args:
#     - key0: PRNG key
#     - x: batch of data samples
#     - EBM_params: energy-based model parameters
#     - GEN_params: generator parameters
#     - EBM_fwd: energy-based model forward pass, --immutable
#     - GEN_fwd: generator forward pass, --immutable
#     - temp_schedule: temperature schedule

#     Returns:
#     - total_loss_ebm: total loss for the EBM model
#     - total_loss_gen: total loss for the GEN model
#     """

#     # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
#     temp_schedule = jnp.concatenate([jnp.array([0]), temp_schedule])

#     def loss(carry, i):
#         key_i, total_loss_ebm, total_loss_gen = carry

#         # Sample from the prior and posterior distributions
#         key_i, z_prior = sample_prior(key_i, EBM_params, EBM_fwd)
#         key_i, z_posterior = sample_posterior(
#             key_i, x, temp_schedule[i], EBM_params, GEN_params, EBM_fwd, GEN_fwd
#         )

#         # Compute the loss for both models
#         loss_current_ebm = ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd)
#         key_i, loss_current_gen = gen_loss(key_i, x, z_posterior, GEN_params, GEN_fwd)

#         # ∇T = t_i - t_{i-1}
#         delta_T = temp_schedule[i] - temp_schedule[i - 1]

#         # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
#         total_loss_ebm += 0.5 * (loss_current_ebm + total_loss_ebm) * delta_T
#         total_loss_gen += 0.5 * (loss_current_gen + total_loss_gen) * delta_T

#         return (key_i, total_loss_ebm, total_loss_gen), None

#     total_loss_ebm = 0
#     total_loss_gen = 0

#     initial_state = (key0, total_loss_ebm, total_loss_gen)

#     (final_key, final_loss_ebm, final_loss_gen), _ = scan(
#         loss, initial_state, jnp.arange(1, len(temp_schedule))
#     )

#     return final_loss_ebm, final_loss_gen


def ThermoEBM_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = jnp.concatenate([jnp.array([0]), temp_schedule])

    def loss(carry, i):
        key_i, total_loss_ebm = carry

        # Sample from the prior and posterior distributions
        key_i, z_prior = sample_prior(key_i, EBM_params, EBM_fwd)
        key_i, z_posterior = sample_posterior(
            key_i, x, temp_schedule[i], EBM_params, GEN_params, EBM_fwd, GEN_fwd
        )

        # Compute the loss for both models
        loss_current_ebm = ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss_ebm += 0.5 * (loss_current_ebm + total_loss_ebm) * delta_T

        return (key_i, total_loss_ebm), None

    total_loss_ebm = 0

    initial_state = (key, total_loss_ebm)

    (_, final_loss_ebm), _ = scan(
        loss, initial_state, jnp.arange(1, len(temp_schedule))
    )

    return final_loss_ebm


def ThermoGEN_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):

    # Prepend 0 to the temperature schedule, for unconditional ∇T calculation
    temp_schedule = jnp.concatenate([jnp.array([0]), temp_schedule])

    def loss(carry, i):
        key_i, total_loss_gen = carry

        # Sample from the prior and posterior distributions
        key_i, z_posterior = sample_posterior(
            key_i, x, temp_schedule[i], EBM_params, GEN_params, EBM_fwd, GEN_fwd
        )

        # Compute the loss for both models
        key_i, loss_current_gen = gen_loss(key_i, x, z_posterior, GEN_params, GEN_fwd)

        # ∇T = t_i - t_{i-1}
        delta_T = temp_schedule[i] - temp_schedule[i - 1]

        # # 1/2 * (f(x_i) + f(x_{i-1})) * ∇T
        total_loss_gen += 0.5 * (loss_current_gen + total_loss_gen) * delta_T

        return (key_i, total_loss_gen), None

    total_loss_gen = 0

    initial_state = (key, total_loss_gen)

    (_, final_loss_gen), _ = scan(
        loss, initial_state, jnp.arange(1, len(temp_schedule))
    )

    return final_loss_gen
