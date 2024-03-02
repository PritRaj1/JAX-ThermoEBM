import jax
import jax.numpy as jnp
from functools import partial
import optax
import configparser

from src.pipeline.loss_fcn import ThermodynamicIntegrationLoss
from src.utils.helper_functions import get_grad_var
from src.MCMC_Samplers.sample_distributions import sample_prior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

TI_loss_batched = jax.vmap(ThermodynamicIntegrationLoss, in_axes=(0, 0, None, None, None, None, None))

@partial(jax.jit, static_argnums=(4,5))
def train_step(key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule):

    # Split the key for batched operations
    key_batch = jax.random.split(key, batch_size + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]

    # Compute loss of both models
    losses = TI_loss_batched(sub_key_batch, x, *params_tup, *fwd_fcn_tup, temp_schedule)

    # Find gradients
    key_batch = jax.random.split(key, batch_size + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]

    # with jax.profiler.trace("/tmp/jax-trace", create_perfetto_link=True):
    Jacob = jax.jacfwd(TI_loss_batched, argnums=(2,3))(sub_key_batch, x, *params_tup, *fwd_fcn_tup, temp_schedule)

    grad_ebm = Jacob[0] # [ [ L_e/dθ1, L_e/dθ2, ... ], [ L_g/dθ1, L_g/dθ2, ... ] ]
    grad_gen = Jacob[1] # [ [ L_e/dΨ1, L_e/dΨ2, ... ], [ L_g/dΨ1, L_g/dΨ2, ... ] ]

    # Get mean gradient across all items in batch, and extract L_e w.r.t θ and L_g w.r.t Ψ
    grad_ebm = jax.tree_map(lambda x: jnp.mean(x, axis=0)[0], grad_ebm)
    grad_gen = jax.tree_map(lambda x: jnp.mean(x, axis=0)[1], grad_gen)

    grad_list = [grad_ebm, grad_gen]
    
    new_opt_states = []
    new_params_set = []

    # Update the parameters
    for i in range(len(optimiser_tup)):
        updates, new_opt_state = optimiser_tup[i].update(grad_list[i], opt_state_tup[i])
        new_params = optax.apply_updates(params_tup[i], updates)
        new_opt_states.append(new_opt_state)
        new_params_set.append(new_params)

    total_loss = losses.mean(axis=0).sum() # L_e + L_g
    grad_var = get_grad_var(*grad_list)

    # print(f"Total Loss: {total_loss}, Grad Var: {grad_var}")

    return (
        key,
        tuple(new_params_set),
        tuple(new_opt_states),
        total_loss,
        grad_var
    )


@partial(jax.jit, static_argnums=(3,))
def validate(key, x, params_tup, fwd_fcn_tup, temp_schedule):

    # Split the key for batched operations
    key_batch = jax.random.split(key, batch_size + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]

    # Compute loss of both models
    losses = TI_loss_batched(sub_key_batch, x, *params_tup, *fwd_fcn_tup, temp_schedule)

    # Find gradients
    key_batch = jax.random.split(key, batch_size + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]
    
    Jacob = jax.jacfwd(TI_loss_batched, argnums=(2,3))(sub_key_batch, x, *params_tup, *fwd_fcn_tup, temp_schedule)
    grad_ebm = Jacob[0] # [ [ L_e/dθ1, L_e/dθ2, ... ], [ L_g/dθ1, L_g/dθ2, ... ] ]
    grad_gen = Jacob[1] # [ [ L_e/dΨ1, L_e/dΨ2, ... ], [ L_g/dΨ1, L_g/dΨ2, ... ] ]

    # Get mean gradient across all items in batch, and extract L_e w.r.t θ and L_g w.r.t Ψ
    grad_ebm = jax.tree_map(lambda x: jnp.mean(x, axis=0)[0], grad_ebm)
    grad_gen = jax.tree_map(lambda x: jnp.mean(x, axis=0)[1], grad_gen)

    total_loss = losses.mean(axis=0).sum() # L_e + L_g
    grad_var = get_grad_var(grad_ebm, grad_gen)

    return key, total_loss, grad_var

@partial(jax.jit, static_argnums=(2)) 
def generate(key, params_tup, fwd_fcn_tup):

    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    x_pred = fwd_fcn_tup[1](params_tup[1], jax.lax.stop_gradient(z))

    return key, x_pred[0]