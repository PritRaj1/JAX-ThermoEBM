
import jax
import jax.numpy as jnp
from functools import partial
from src.MCMC_Samplers.grad_log_probs import prior_grad_log, posterior_grad_log

def update_step(key, x, grad_f, s):
    """Update the current state of the sampler."""
    x += (s * grad_f)

    key, subkey = jax.random.split(key)
    x += jnp.sqrt(2 * s) * jax.random.normal(subkey, x.shape)

    return key, x

@partial(jax.jit, static_argnums=(1,2,3))
def sample_p0(key, p0_sig, batch_size, num_z):
    """Sample from the prior distribution."""
    
    key, subkey = jax.random.split(key)
    return key, p0_sig * jax.random.normal(subkey, (batch_size, num_z, 1, 1))

@partial(jax.jit, static_argnums=(1,3,4,5,6,7))
def sample_prior(key, 
                 EBM_fwd, 
                 EBM_params, 
                 p0_sig, 
                 step_size, 
                 num_steps, 
                 batch_size, 
                 num_z):
    """
    Sample from the prior distribution.
    
    Args:
    - key: PRNG key
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - p0_sig: prior sigma, --immutable
    - step_size: step size, --immutable
    - num_steps: number of steps, --immutable
    - batch_size: batch size, --immutable
    - num_z: number of latent space variables, --immutable

    Returns:
    - key: PRNG key
    - z: latent space variable sampled from p_a(x)
    """
    
    key, z = sample_p0(key, p0_sig, batch_size, num_z)
    
    for k in range(num_steps):
        grad_f = prior_grad_log(z, EBM_fwd, EBM_params, p0_sig)
        key, z = update_step(key, z, grad_f, step_size)

    return key, z

@partial(jax.jit, static_argnums=(2,4,6,7,8,9,10,12))
def sample_posterior(key, 
                     data,
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
                     temp_schedule):
    """
    Sample from the posterior distribution.
    
    Args:
    - key: PRNG key
    - data: batch of data samples
    - EBM_fwd: energy-based model forward pass, --immutable
    - EBM_params: energy-based model parameters
    - GEN_fwd: generator forward pass, --immutable
    - GEN_params: generator parameters
    - pl_sig: likelihood sigma, --immutable
    - p0_sig: prior sigma, --immutable
    - step_size: step size, --immutable
    - num_steps: number of steps, --immutable
    - batch_size: batch size, --immutable
    - num_z: number of latent space variables, --immutable
    - temp_schedule: temperature schedule, --immutable
    
    Returns:
    - key: PRNG key
    - z_samples: samples from the posterior distribution indexed by temperature
    """
    
    z_samples = jnp.zeros((len(temp_schedule), batch_size, num_z, 1, 1))

    for idx, t in enumerate(temp_schedule):
        key, z = sample_p0(key, p0_sig, batch_size, num_z)
        
        for k in range(num_steps):
            grad_f = posterior_grad_log(z, data, t, EBM_fwd, EBM_params, GEN_fwd, GEN_params, pl_sig, p0_sig)
            key, z = update_step(key, z, grad_f, step_size)
        
        z_samples = jax.ops.index_update(z_samples, idx, z)

    return key, z_samples