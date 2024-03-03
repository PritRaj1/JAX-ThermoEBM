import jax
from jax import value_and_grad
import jax.numpy as jnp
from functools import partial
import optax

from src.MCMC_Samplers.sample_distributions import sample_prior
from src.pipeline.loss_computation.loss_fcn import ThermoEBM_loss, ThermoGEN_loss


def get_losses_and_grads(key, x, params_tup, fwd_fcn_tup, temp_schedule):
    """
        Function to compute the losses and gradients of the models.

        Args:
        - key: PRNG key
        - x: one item from the dataset
        - params_tup: tuple of model parameters
        - fwd_fcn_tup: tuple of model forward passes
        - temp_schedule: temperature schedule

        Returns:
        - loss_ebm: EBM loss
        - grad_ebm: EBM gradients
        - loss_gen: GEN loss
        - grad_gen: GEN gradients

    *Note: The same key is utilized to maintain consistency 
    in z_posterior across the two Thermodynamic Integration loops. 
    These loops are separated to circumvent the necessity of computing 
    a Jacobian when dealing with two output losses simultaneously.
    """

    # Compute loss of both models, use the same key to ensure z_posterior is the same
    loss_ebm, grad_ebm = value_and_grad(ThermoEBM_loss, argnums=2)(
        key, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )
    loss_gen, grad_gen = value_and_grad(ThermoGEN_loss, argnums=3)(
        key, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )

    return loss_ebm, grad_ebm, loss_gen, grad_gen


def update_params(optimiser_tup, grad_list, opt_state_tup, params_tup):
    """
    Function to update the parameters of the models.

    Args:
    - optimiser_tup: tuple of optimisers
    - grad_list: list of gradients
    - opt_state_tup: tuple of optimiser states
    - params_tup: tuple of model parameters

    Returns:
    - new_params_set: tuple of updated model parameters
    - new_opt_states: tuple of updated optimiser states
    """

    ebm_updates, new_ebm_opt_state = optimiser_tup[0].update(
        grad_list[0], opt_state_tup[0]
    )

    gen_updates, new_gen_opt_state = optimiser_tup[1].update(
        grad_list[1], opt_state_tup[1]
    )

    new_ebm_params = optax.apply_updates(params_tup[0], ebm_updates)
    new_gen_params = optax.apply_updates(params_tup[1], gen_updates)

    return (new_ebm_params, new_gen_params), (new_ebm_opt_state, new_gen_opt_state)


def generate(key, params_tup, fwd_fcn_tup):
    """Generates a single image from the generator."""
    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    x_pred = fwd_fcn_tup[1](params_tup[1], jax.lax.stop_gradient(z))

    return key, x_pred


def get_grad_var(grad_ebm, grad_gen):

    # Get gradients from grad dictionaries
    grad_ebm = jax.tree_util.tree_flatten(grad_ebm)[0]
    grad_gen = jax.tree_util.tree_flatten(grad_gen)[0]

    # Flatten the gradients
    grad_ebm = jnp.concatenate([jnp.ravel(g) for g in grad_ebm])
    grad_gen = jnp.concatenate([jnp.ravel(g) for g in grad_gen])

    return jnp.var(jnp.concatenate([grad_ebm, grad_gen]))
