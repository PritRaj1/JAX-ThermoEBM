import jax
import jax.numpy as jnp

from src.pipeline.loss_fcn import TI_EBM_loss_fcn, TI_GEN_loss_fcn

def EBM_loss_fcn_batched(key, x_batch, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):

    def EBM_loss_fcn(one_key, x):
        return TI_EBM_loss_fcn(one_key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)
    
    # Vectorize the computation over the batch dimension
    batch_loss, batch_key = jax.vmap(EBM_loss_fcn, in_axes=(0,0))(jax.random.split(key, x_batch.shape[0]), x_batch)

    # Return batch_loss and batch_key
    return jnp.mean(batch_loss), batch_key[-1]

def GEN_loss_fcn_batched(key, x_batch, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule):

    def GEN_loss_fcn(one_key, x):
        return TI_GEN_loss_fcn(one_key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)
    
    # Vectorize the computation over the batch dimension
    batch_loss, batch_key = jax.vmap(GEN_loss_fcn, in_axes=(0,0))(jax.random.split(key, x_batch.shape[0]), x_batch)

    # Return batch_loss and batch_key
    return jnp.mean(batch_loss), batch_key[-1]