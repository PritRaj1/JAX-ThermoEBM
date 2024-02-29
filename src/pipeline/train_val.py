import jax
import jax.numpy as jnp
from jax import value_and_grad
from functools import partial
import optax

from src.pipeline.batched_loss_fcns import EBM_loss_fcn_batched, GEN_loss_fcn_batched
from src.utils.helper_functions import get_grad_var
from src.MCMC_Samplers.sample_distributions import sample_prior

@partial(jax.jit, static_argnums=(4,5,6))
def train_step(key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule):

    (loss_ebm, ebm_key), grad_ebm = value_and_grad(EBM_loss_fcn_batched, argnums=2, has_aux=True)(key, x, *params_tup, *fwd_fcn_tup, temp_schedule)
    (loss_gen, gen_key), grad_gen = value_and_grad(GEN_loss_fcn_batched, argnums=3, has_aux=True)(ebm_key, x, *params_tup, *fwd_fcn_tup, temp_schedule)

    # Get mean gradient to update the parameters
    mean_grad_ebm = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_ebm)
    mean_grad_gen = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_gen)

    mean_grad_list = [mean_grad_ebm, mean_grad_gen]
    
    new_opt_states = []
    new_params_set = []

    # Update the parameters
    for i in range(len(optimiser_tup)):
        updates, new_opt_state = optimiser_tup[i].update(mean_grad_list[i], opt_state_tup[i])
        new_params = optax.apply_updates(params_tup[i], updates)
        new_opt_states.append(new_opt_state)
        new_params_set.append(new_params)

    total_loss = loss_ebm.mean() + loss_gen.mean()
    grad_var = get_grad_var(mean_grad_ebm, mean_grad_gen)

    return (
        gen_key,
        tuple(new_params_set),
        tuple(new_opt_states),
        total_loss,
        grad_var
    )


@partial(jax.jit, static_argnums=(3,4))
def validate(key, x, params_tup, fwd_fcn_tup, temp_schedule):
    (loss_ebm, ebm_key), grad_ebm = value_and_grad(EBM_loss_fcn_batched, argnums=2, has_aux=True)(key, x, *params_tup, *fwd_fcn_tup, temp_schedule)

    (loss_gen, gen_key), grad_gen = value_and_grad(GEN_loss_fcn_batched, argnums=3, has_aux=True)(ebm_key, x, *params_tup, *fwd_fcn_tup, temp_schedule)

    # Get mean gradient to update the parameters
    mean_grad_ebm = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_ebm)
    mean_grad_gen = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_gen)

    total_loss = loss_ebm.mean() + loss_gen.mean()
    grad_var = get_grad_var(mean_grad_ebm, mean_grad_gen)

    return gen_key, total_loss, grad_var

# @partial(jax.jit, static_argnums=(2)) 
def generate(key, params_tup, fwd_fcn_tup):
    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    x_pred = fwd_fcn_tup[1](params_tup[1], jax.lax.stop_gradient(z))

    return x_pred[0]