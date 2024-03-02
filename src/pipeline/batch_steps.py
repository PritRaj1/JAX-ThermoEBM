import jax
import optax
from functools import partial
from jax import value_and_grad
import configparser
import numpy as np
import matplotlib.pyplot as plt

from src.pipeline.update_steps import *
from src.MCMC_Samplers.sample_distributions import sample_prior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

batch_loss_grad = jax.vmap(get_losses_and_grads, in_axes=(0, 0, None, None, None))

@partial(jax.jit, static_argnums=(4, 5))
def train_step(
    key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule
):
    
    # Get a batch of keys
    key_batch = jax.random.split(key, batch_size + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]

    # Compute loss of both models
    batch_loss_e, batch_grad_e, batch_loss_g, batch_grad_g = batch_loss_grad(
        sub_key_batch, x, params_tup, fwd_fcn_tup, temp_schedule
    )

    batch_grad_list = [batch_grad_e, batch_grad_g]
    new_opt_states = []
    new_params_set = []

    # Update the parameters
    params_tup, opt_state_tup = update_params(
        optimiser_tup, batch_grad_list, opt_state_tup, params_tup
    )

    total_loss = batch_loss_e.mean() + batch_loss_g.mean()  # L_e + L_g
    grad_var = get_grad_var(*batch_grad_list)

    # print(f"Total Loss: {total_loss}, Grad Var: {grad_var}")

    return key, params_tup, opt_state_tup, total_loss, grad_var


@partial(jax.jit, static_argnums=(3,))
def validate(key, x, params_tup, fwd_fcn_tup, temp_schedule):

    # Get a batch of keys
    key_batch = jax.random.split(key, batch_size + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]

    # Compute loss of both models
    batch_loss_e, batch_grad_e, batch_loss_g, batch_grad_g = batch_loss_grad(
        sub_key_batch, x, params_tup, fwd_fcn_tup, temp_schedule
    )

    total_loss = batch_loss_e.mean() + batch_loss_g.mean()  # L_e + L_g
    grad_var = get_grad_var(batch_grad_e, batch_grad_g)

    return key, total_loss, grad_var