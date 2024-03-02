import jax
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

    @jax.jit
    def train_batch(carry, idx):
        key, params_old, opt_state_old = carry
        x, _ = next(iter(train_loader))
        key, params_new, opt_state_new, loss, grad_var = train_step(
            key, x, params_old, opt_state_old, optimiser_tup, fwd_fcn_tup, temp_schedule
        )
        return (key, params_new, opt_state_new), (loss, grad_var)

    initial_state_train = (key, initial_params_tup, initial_opt_state_tup)

    (final_key, final_params, final_opt_state), (losses, grads) = lax.scan(
        f=train_batch, init=initial_state_train, xs=None, length=len(train_loader)
    )

    return final_key, final_params, final_opt_state, jnp.sum(losses), jnp.sum(grads)


def val_epoch(init_key, val_loader, params_tup, fwd_fcn_tup, temp_schedule):

    #@jax.jit
    def val_batch(carry, idx):
        key = carry
        x, _ = next(iter(val_loader))
        key, loss, grad_var = validate(key, x, params_tup, fwd_fcn_tup, temp_schedule)

        return key, (loss, grad_var)

    final_key, (losses, grads) = lax.scan(
        f=val_batch, init=init_key, xs=None, length=len(val_loader)
    )

    return final_key, jnp.sum(losses), jnp.sum(grads)
