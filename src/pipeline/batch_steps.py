import jax
import configparser

from src.pipeline.update_steps import *

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

batch_compute = jax.vmap(get_losses_and_grads, in_axes=(0, 0, None, None))

def train_step(
    key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup
):
    """Single batch of parameter updates for the EBM and GEN models."""

    # Compute loss and grad of each batch
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    batch_loss_e, batch_grad_e, batch_loss_g, batch_grad_g = batch_compute(
        subkey_batch, x, params_tup, fwd_fcn_tup
    )

    # Take sum across batch, (reduction = sum, as used in Pang et al.)
    grad_ebm = jax.tree_util.tree_map(lambda x: x.sum(0), batch_grad_e)
    grad_gen = jax.tree_util.tree_map(lambda x: x.sum(0), batch_grad_g)
    grad_list = [grad_ebm, grad_gen]

    # Update the parameters
    params_tup, opt_state_tup = update_params(
        optimiser_tup, grad_list, opt_state_tup, params_tup
    )

    total_loss = batch_loss_e.sum() + batch_loss_g.sum()  # L_e + L_g
    grad_var = get_grad_var(*grad_list)

    return key, params_tup, opt_state_tup, total_loss, grad_var


def val_step(key, x, params_tup, fwd_fcn_tup):
    """Single batch evaluation on an unseen image set."""

    # Compute loss of each batch
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    batch_loss_e, batch_grad_e, batch_loss_g, batch_grad_g = batch_compute(
        subkey_batch, x, params_tup, fwd_fcn_tup
    )

    # Take sum across batch, (reduction = sum, as used in Pang et al.)
    grad_ebm = jax.tree_util.tree_map(lambda x: x.sum(0), batch_grad_e)
    grad_gen = jax.tree_util.tree_map(lambda x: x.sum(0), batch_grad_g)

    total_loss = batch_loss_e.sum() + batch_loss_g.sum()  # L_e + L_g
    grad_var = get_grad_var(grad_ebm, grad_gen)

    return key, total_loss, grad_var
