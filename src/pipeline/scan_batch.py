import jax.numpy as jnp
from jax import lax

from src.pipeline.batch_steps import train_step, validate


def train_epoch(
    key,
    train_loader,
    initial_params_tup,
    initial_opt_state_tup,
    optimiser_tup,
    fwd_fcn_tup,
    temp_schedule,
):
    """
    Function to train the model for one epoch.

    Args:
    - key: PRNG key
    - train_loader: training data loader
    - initial_params_tup: initial model parameters and epoch start
    - initial_opt_state_tup: initial optimiser states
    - optimiser_tup: tuple of optimisers
    - fwd_fcn_tup: tuple of model forward passes
    - temp_schedule: temperature schedule
    """

    def train_batch(carry, idx):
        key, params_old, opt_state_old, train_loss, train_grad_var = carry
        x, _ = next(iter(train_loader))
        key, params_new, opt_state_new, loss, grad_var = train_step(
            key, x, params_old, opt_state_old, optimiser_tup, fwd_fcn_tup, temp_schedule
        )
        train_loss += loss
        train_grad_var += grad_var
        return (key, params_new, opt_state_new, train_loss, train_grad_var), None

    initial_state_train = (key, initial_params_tup, initial_opt_state_tup, 0, 0)

    (final_key, final_params, final_opt_state, train_loss, train_grad_var), _ = lax.scan(
        train_batch, initial_state_train, jnp.arange(len(train_loader))
    )

    return final_key, final_params, final_opt_state, train_loss, train_grad_var


def val_epoch(key, val_loader, params_tup, fwd_fcn_tup, temp_schedule):

    def val_batch(carry, idx):
        key, val_loss, val_grad_var = carry
        x, _ = next(iter(val_loader))
        key, loss, grad_var = validate(key, x, params_tup, fwd_fcn_tup, temp_schedule)
        val_loss += loss
        val_grad_var += grad_var
        return (key, val_loss, val_grad_var), None

    initial_state_val = (key, 0, 0)

    (final_key, val_loss, val_grad_var), _ = lax.scan(
        val_batch, initial_state_val, jnp.arange(len(val_loader))
    )

    return final_key, val_loss, val_grad_var
