import jax
from src.MCMC_Samplers.sample_distributions import sample_prior

def generate_one(key, params_tup, fwd_fcn_tup):
        """Generates a single image from the generator."""
        key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
        x_pred = fwd_fcn_tup[1](params_tup[1], jax.lax.stop_gradient(z))

        return x_pred

generate_batch = jax.vmap(generate_one, in_axes=(0, None, None))

def generate(key, params_tup, num_images, fwd_fcn_tup):
    """Generates multiple images from the generator."""

    # Create a batch of keys
    key_batch = jax.random.split(key, num_images + 1)
    key, sub_key_batch = key_batch[0], key_batch[1:]

    # Generate 'num_images' samples
    x_pred = generate_batch(sub_key_batch, params_tup, fwd_fcn_tup)

    return key, x_pred