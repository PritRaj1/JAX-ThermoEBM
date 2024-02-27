import jax
import jax.numpy as jnp

class Langevin_Sampler():
    def __init__(self, step_size, num_steps):
        self.s = step_size
        self.K = num_steps
        self.grad_log_fcn = None

    def __call__(self, x, state, key, data=None):
        """
        Estimate a sample from the target distribution using Langevin MCMC.
        """
        for k in range(self.K):
            
            # Deterministic Gradient Step
            x += (self.s * self.grad_log_fcn(x, state, data))

            # Stochastic Noise Step
            key, subkey = jax.random.split(key)
            x += jnp.sqrt(2 * self.s) * jax.random.normal(subkey, x.shape)

        return x