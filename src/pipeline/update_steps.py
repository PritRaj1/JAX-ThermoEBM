import jax
from jax import value_and_grad
import jax.numpy as jnp
import optax
import configparser
from jax.lax import stop_gradient

from src.pipeline.loss_computation.loss_fcn import (
    Thermo_loss,
    vanilla_EBM_loss,
    vanilla_GEN_loss,
)

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

temp_power = int(parser["TEMP"]["TEMP_POWER"])

if temp_power == 0:
    loss_EBM = vanilla_EBM_loss
    loss_GEN = vanilla_GEN_loss
    kill_duplicate = 1
else:
    loss_EBM = Thermo_loss
    loss_GEN = Thermo_loss
    kill_duplicate = 0 # This is used to kill the duplicate loss evaluation in the Thermo_loss function


def get_losses_and_grads(key, x, params_tup, fwd_fcn_tup):
    """
    Function to compute the losses and gradients of the models.

    Args:
    - key: PRNG key
    - x: batch of data
    - params_tup: tuple of model parameters
    - fwd_fcn_tup: tuple of model forward passes

    Returns:
    - loss_ebm: EBM loss
    - grad_ebm: EBM gradients
    - loss_gen: GEN loss
    - grad_gen: GEN gradients

    *Note: The same key is used for each loss to maintain consistency
    in z_posterior across the two loss computations.
    These computations are separated to circumvent the necessity of computing
    a Jacobian when dealing with two output losses simultaneously.
    """

    ebm_params, gen_params = params_tup

    # Compute loss of both models, use the same key to ensure z_posterior is the same
    loss_ebm, grad_ebm = value_and_grad(loss_EBM, argnums=2)(
        key, x, ebm_params, stop_gradient(gen_params), *fwd_fcn_tup
    )
    loss_gen, grad_gen = value_and_grad(loss_GEN, argnums=3)(
        key, x, stop_gradient(ebm_params), gen_params, *fwd_fcn_tup
    )

    return loss_ebm * kill_duplicate, grad_ebm, loss_gen, grad_gen


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


def get_grad_var(grad_ebm, grad_gen):

    # Get gradients from grad dictionaries
    grad_ebm = jax.tree_util.tree_flatten(grad_ebm)[0]
    grad_gen = jax.tree_util.tree_flatten(grad_gen)[0]

    # Flatten the gradients
    grad_ebm = jnp.concatenate([jnp.ravel(g) for g in grad_ebm])
    grad_gen = jnp.concatenate([jnp.ravel(g) for g in grad_gen])

    return jnp.var(jnp.concatenate([grad_ebm, grad_gen]))
