import jax
from jax import value_and_grad
import optax
from functools import partial

from src.pipeline.loss_fcn import TI_EBM_loss_fcn, TI_GEN_loss_fcn


@partial(jax.jit, static_argnums=(2, 4, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15))
def get_losses(
    key,
    x,
    EBM_fwd,
    EBM_params,
    GEN_fwd,
    GEN_params,
    pl_sig,
    p0_sig,
    EBM_step_size,
    EBM_num_steps,
    GEN_step_size,
    GEN_num_steps,
    batch_size,
    z_channels,
    temp_schedule,
):

    def get_losses_grads(x):
        (loss_ebm, ebm_key), grad_ebm = value_and_grad(
            TI_EBM_loss_fcn, argnums=3, has_aux=True
        )(
            key,
            x,
            EBM_fwd,
            EBM_params,
            GEN_fwd,
            GEN_params,
            pl_sig,
            p0_sig,
            EBM_step_size,
            EBM_num_steps,
            batch_size,
            z_channels,
            temp_schedule,
        )

        (loss_gen, gen_key), grad_gen = value_and_grad(
            TI_GEN_loss_fcn, argnums=5, has_aux=True
        )(
            ebm_key,
            x,
            EBM_fwd,
            EBM_params,
            GEN_fwd,
            GEN_params,
            pl_sig,
            p0_sig,
            GEN_step_size,
            GEN_num_steps,
            batch_size,
            z_channels,
            temp_schedule,
        )

        return gen_key, loss_ebm, grad_ebm, loss_gen, grad_gen

    key, loss_ebm, grad_ebm, loss_gen, grad_gen = jax.vmap(get_losses_grads)(x)

    return key, loss_ebm, grad_ebm, loss_gen, grad_gen


@partial(jax.jit, static_argnums=(6, 7))
def update_params(
    EBM_params,
    EBM_opt_state,
    GEN_params,
    GEN_opt_state,
    grad_ebm,
    grad_gen,
    EBM_optimiser,
    GEN_optimiser,
):

    updates, EBM_opt_state = EBM_optimiser.update(grad_ebm, EBM_opt_state)
    EBM_params = optax.apply_updates(EBM_params, updates)
    updates, GEN_opt_state = GEN_optimiser.update(grad_gen, GEN_opt_state)
    GEN_params = optax.apply_updates(GEN_params, updates)

    return EBM_params, EBM_opt_state, GEN_params, GEN_opt_state
