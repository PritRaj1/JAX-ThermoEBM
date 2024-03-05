import jax
import jax.numpy as jnp
from functools import partial
from src.MCMC_Samplers.sample_distributions import sample_prior
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sigma = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
image_dim = 64 if parser["PIPELINE"]["DATASET"] == "CelebA" else 32


@partial(jax.jit, static_argnums=3)
def generate(key, _, params_tup, fwd_fcn_tup):
    """Generate a batch image from the generator."""

    key, z = sample_prior(key, params_tup[0], fwd_fcn_tup[0])
    key, subkey = jax.random.split(key)
    x_gen = fwd_fcn_tup[1](params_tup[1], z) + (
        pl_sigma * jax.random.normal(subkey, (batch_size, image_dim, image_dim, 3))
    )

    return key, x_gen


def generate_images(key, params_tup, num_images, fwd_fcn_tup):
    """Generates 'num_images' images from the generator."""

    # The generator struggles with memory, so generate them in batches
    scan_gen = partial(generate, params_tup=params_tup, fwd_fcn_tup=fwd_fcn_tup)
    key, images = jax.lax.scan(
        f=scan_gen, init=key, xs=None, length=num_images // batch_size
    )
    images = jnp.reshape(images, (-1, image_dim, image_dim, 3))

    return key, images
