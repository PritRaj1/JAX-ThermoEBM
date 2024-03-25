import jax
import jax.numpy as jnp
import optax
import configparser
from src.pipeline.loss_computation.loss_and_grad import vanilla_computation, thermo_computation

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

temp_power = int(parser["TEMP"]["TEMP_POWER"])

if temp_power == 0:
    loss_computation = vanilla_computation
else:
    loss_computation = thermo_computation

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

    return loss_computation(key, x, params_tup, fwd_fcn_tup)


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
