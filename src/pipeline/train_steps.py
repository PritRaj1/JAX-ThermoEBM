import jax
from jax import value_and_grad
import optax

from src.pipeline.loss_fcn import TI_EBM_loss_fcn, TI_GEN_loss_fcn

@jax.jit
def train_ebm_step(key, 
                   x, 
                   EBM_fwd, 
                   EBM_params, 
                   GEN_fwd, 
                   GEN_params, 
                   pl_sig, 
                   p0_sig, 
                   step_size, 
                   num_steps, 
                   batch_size, 
                   num_z, 
                   temp_schedule, 
                   EBM_optimiser, 
                   EBM_opt_state):

    (loss_ebm, key), grad_ebm = value_and_grad(TI_EBM_loss_fcn, argnums=3, has_aux=True)(key, 
                                                                                        x,
                                                                                        EBM_fwd,
                                                                                        EBM_params,
                                                                                        GEN_fwd,
                                                                                        GEN_params,
                                                                                        pl_sig,
                                                                                        p0_sig,
                                                                                        step_size,
                                                                                        num_steps,
                                                                                        batch_size,
                                                                                        num_z,
                                                                                        temp_schedule)
    
    updates, EBM_opt_state = EBM_optimiser.update(grad_ebm, EBM_opt_state)
    EBM_params = optax.apply_updates(EBM_params, updates)

    return key, EBM_params, EBM_opt_state, loss_ebm, grad_ebm

@jax.jit
def train_gen_step(key,
                     x, 
                     EBM_fwd, 
                     EBM_params, 
                     GEN_fwd, 
                     GEN_params, 
                     pl_sig, 
                     p0_sig, 
                     step_size, 
                     num_steps, 
                     batch_size, 
                     num_z, 
                     temp_schedule, 
                     GEN_optimiser, 
                     GEN_opt_state):
    
     (loss_gen, key), grad_gen = value_and_grad(TI_GEN_loss_fcn, argnums=5, has_aux=True)(key, 
                                                                                         x,
                                                                                         EBM_fwd,
                                                                                         EBM_params,
                                                                                         GEN_fwd,
                                                                                         GEN_params,
                                                                                         pl_sig,
                                                                                         p0_sig,
                                                                                         step_size,
                                                                                         num_steps,
                                                                                         batch_size,
                                                                                         num_z,
                                                                                         temp_schedule)
     
     updates, GEN_opt_state = GEN_optimiser.update(grad_gen, GEN_opt_state)
     GEN_params = optax.apply_updates(GEN_params, updates)
    
     return key, GEN_params, GEN_opt_state, loss_gen, grad_gen


        