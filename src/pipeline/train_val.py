import jax
import optax
from functools import partial
from jax import value_and_grad
import configparser

from src.pipeline.loss_fcn import ThermoEBM_loss, ThermoGEN_loss
from src.utils.helper_functions import get_grad_var
from src.MCMC_Samplers.sample_distributions import sample_prior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])


@partial(jax.jit, static_argnums=(4, 5))
def train_step(
    key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule
):

    # Use same subkey for both models to ensure same z is sampled
    key, subkey = jax.random.split(key)

    # Compute loss of both models
    loss_ebm, grad_ebm = value_and_grad(ThermoEBM_loss, argnums=2)(
        subkey, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )
    loss_gen, grad_gen = value_and_grad(ThermoGEN_loss, argnums=3)(
        subkey, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )

    grad_list = [grad_ebm, grad_gen]
    new_opt_states = []
    new_params_set = []

    # Update the parameters
    for i in range(len(optimiser_tup)):
        updates, new_opt_state = optimiser_tup[i].update(grad_list[i], opt_state_tup[i])
        new_params = optax.apply_updates(params_tup[i], updates)
        new_opt_states.append(new_opt_state)
        new_params_set.append(new_params)

    total_loss = loss_ebm + loss_gen  # L_e + L_g
    grad_var = get_grad_var(*grad_list)

    # print(f"Total Loss: {total_loss}, Grad Var: {grad_var}")

    return (key, tuple(new_params_set), tuple(new_opt_states), total_loss, grad_var)


@partial(jax.jit, static_argnums=(3,))
def validate(key, x, params_tup, fwd_fcn_tup, temp_schedule):

    key, subkey = jax.random.split(key)

    # Compute loss of both models
    loss_ebm, grad_ebm = value_and_grad(ThermoEBM_loss, argnums=2)(
        subkey, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )
    loss_gen, grad_gen = value_and_grad(ThermoGEN_loss, argnums=3)(
        subkey, x, *params_tup, *fwd_fcn_tup, temp_schedule
    )

    total_loss = loss_ebm + loss_gen  # L_e + L_g
    grad_var = get_grad_var(grad_ebm, grad_gen)

    return key, total_loss, grad_var


def generate(key, params_tup, fwd_fcn_tup):

    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    x_pred = fwd_fcn_tup[1](params_tup[1], jax.lax.stop_gradient(z))

    return key, x_pred[0]
