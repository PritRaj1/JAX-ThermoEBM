
from jax import value_and_grad
from jax.lax import stop_gradient

from src.pipeline.loss_computation.loss_fcn import (
    Thermo_loss,
    vanilla_EBM_loss,
    vanilla_GEN_loss,
)

def vanilla_computation(key, x, params_tup, fwd_fcn_tup):
    """Function to compute the losses and gradients of the vanilla models."""

    ebm_params, gen_params = params_tup

    # Compute loss of both models
    loss_ebm, grad_ebm = value_and_grad(vanilla_EBM_loss, argnums=2)(
        key, stop_gradient(x), ebm_params, stop_gradient(gen_params), *fwd_fcn_tup
    )
    loss_gen, grad_gen = value_and_grad(vanilla_GEN_loss, argnums=3)(
        key, stop_gradient(x), stop_gradient(ebm_params), gen_params, *fwd_fcn_tup
    )

    return loss_ebm, grad_ebm, loss_gen, grad_gen

def thermo_computation(key, x, params_tup, fwd_fcn_tup):
    """Function to compute the losses and gradients of the thermo-integration models."""

    ebm_params, gen_params = params_tup

    # Compute loss of both models, use the same key to ensure z_posterior is the same
    loss_total, (grad_ebm, grad_gen) = value_and_grad(Thermo_loss, argnums=(2, 3))(
        key, stop_gradient(x), ebm_params, gen_params, *fwd_fcn_tup
    )

    return 0, grad_ebm, loss_total, grad_gen