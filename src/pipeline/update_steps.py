import jax
from jax import value_and_grad
import jax.numpy as jnp
from functools import partial
import optax

from src.MCMC_Samplers.sample_distributions import sample_prior
from src.pipeline.loss_fcn import ThermoEBM_loss, ThermoGEN_loss


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
    """

    # Compute loss of both models
    loss_ebm, grad_ebm = value_and_grad(ThermoEBM_loss, argnums=2)(
        key, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )
    loss_gen, grad_gen = value_and_grad(ThermoGEN_loss, argnums=3)(
        key, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )

    return loss_ebm, grad_ebm, loss_gen, grad_gen


def update_params(optimiser_tup, batch_grad_list, opt_state_tup, params_tup):
    """
    Function to update the parameters of the models.

    Args:
    - optimiser_tup: tuple of optimisers
    - batch_grad_list: list of batch gradients
    - opt_state_tup: tuple of optimiser states
    - params_tup: tuple of model parameters

    Returns:
    - new_params_set: tuple of updated model parameters
    - new_opt_states: tuple of updated optimiser states
    """

    new_opt_states = []
    new_params_set = []

    # Update the parameters
    for i in range(len(optimiser_tup)):
        grad = jax.tree_util.tree_map(lambda x: x.mean(0), batch_grad_list[i])
        updates, new_opt_state = optimiser_tup[i].update(grad, opt_state_tup[i])
        new_params = optax.apply_updates(params_tup[i], updates)
        new_opt_states.append(new_opt_state)
        new_params_set.append(new_params)

    return tuple(new_params_set), tuple(new_opt_states)


def generate(key, params_tup, fwd_fcn_tup):
    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    x_pred = fwd_fcn_tup[1](params_tup[1], jax.lax.stop_gradient(z))

    return key, x_pred[0]


def get_grad_var(batch_grad_ebm, batch_grad_gen):

    # Take mean across batch
    grad_ebm = jax.tree_util.tree_map(lambda x: x.mean(0), batch_grad_ebm)
    grad_gen = jax.tree_util.tree_map(lambda x: x.mean(0), batch_grad_gen)

    # Get gradients from grad dictionaries
    grad_ebm = jax.tree_util.tree_flatten(grad_ebm)[0]
    grad_gen = jax.tree_util.tree_flatten(grad_gen)[0]

    # Flatten the gradients
    grad_ebm = jnp.concatenate([jnp.ravel(g) for g in grad_ebm])
    grad_gen = jnp.concatenate([jnp.ravel(g) for g in grad_gen])

    return jnp.var(jnp.concatenate([grad_ebm, grad_gen]))
