import jax
from jax import value_and_grad

from src.pipeline.loss_fcn import TI_EBM_loss_fcn, TI_GEN_loss_fcn

@jax.jit
def val_ebm_step(key, 
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
    
    return key, loss_ebm, grad_ebm

@jax.jit
def val_gen_step(key,
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
    
     return key, loss_gen, grad_gen