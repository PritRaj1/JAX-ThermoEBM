from src.MCMC_Samplers.Langevin_MCMC import Langevin_Sampler

import jax
import jax.numpy as jnp

class posterior_sampler(Langevin_Sampler):
    def __init__(self, step_size, num_steps, lkhood_sigma):
        super().__init__(step_size, num_steps)
        self.lkhood_sig = lkhood_sigma

    def grad_log_fcn(self, z, state, data):
        """
        Function to compute the gradient of the log posterior: log[ p(x | z)^t * p(z) ] w.r.t. z.

        Args:
        - z: latent space variable sampled from p0
        - state: current train state of the model
        - data: batch of data samples

        Returns:
        - ∇_z( log[p_θ(z | x)] ) ∝ ∇_z( log[p(x | z)^t * p(z)] )
        """
        GEN_fwd = state.model_apply['GEN_apply']
        GEN_params = state.params['GEN_params']

        t = state.temp['current']

        def log_llood_fcn(z, data):
            g_z = GEN_fwd(GEN_params, z)
            return - t * (jnp.linalg.norm(data-g_z, axis=-1)**2) / (2.0 * self.lkhood_sig**2)

        # Find the gradient of the log likelihood w.r.t. z
        grad_log_llood = jax.grad(log_llood_fcn)(z, data)[0]

        # Find the gradient of the log prior w.r.t. z
        prior_sampler = state.samplers['prior']
        grad_prior = prior_sampler.grad_log_fcn(z, state)

        return grad_log_llood + grad_prior


