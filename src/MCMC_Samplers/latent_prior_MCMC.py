from src.MCMC_Samplers.Langevin_MCMC import Langevin_Sampler

import jax
import jax.numpy as jnp

class prior_sampler(Langevin_Sampler):
    def __init__(self, step_size, num_steps, p0_sigma, num_z, batch_size):
        super().__init__(step_size, num_steps)
        self.p0_sig = p0_sigma
        self.num_z = num_z
        self.batch_size = batch_size

    def grad_log_fcn(self, z, state, data=None):
        """
        Function to compute the gradient of the log prior: log[p_a(x)] w.r.t. z.
        
        Args:
        - z: latent space variable sampled from p0
        - state: current train state of the model

        Returns:
        - ∇_z( log[p_a(x)] )
        """
        EBM_fwd = state.model_apply['EBM_apply']
        EBM_params = state.params['EBM_params']

        # Find the gradient of the f_a(z) w.r.t. z
        grad_f = jax.grad(EBM_fwd)(EBM_params, z)[1]

        return grad_f - (z / (self.p0_sig**2)) 
    
    def sample_p0(self, key):
        """Sample from simple prior distribution, p0(z) = N(0, p0_sigma)"""

        key, subkey = jax.random.split(key)
        return self.p0_sig * jax.random.normal(subkey, (self.batch_size, self.num_z, 1, 1))


