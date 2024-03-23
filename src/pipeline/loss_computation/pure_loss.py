import jax
import jax.numpy as jnp
import optax
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])

def ebm_loss(z_prior, z_posterior, EBM_params, EBM_fwd):
    """Function to compute energy-difference loss for the EBM model."""

    # Compute the energy of the posterior sample
    en_pos = EBM_fwd(EBM_params, z_posterior).mean()

    # Compute the energy of the prior sample
    en_neg = EBM_fwd(EBM_params, z_prior).mean()

    return (en_pos - en_neg)


def gen_loss(key, x, z, GEN_params, GEN_fwd):
    """Function to compute MSE loss for the GEN model."""

    # Generate a sample from the generator
    key, subkey = jax.random.split(key)
    x_pred = GEN_fwd(GEN_params, z) + (
        pl_sig * jax.random.normal(subkey, x.shape)
    )

    # Compute -log[ p_β(x | z) ] = 1/2 * (x - g(z))^2 / σ^2
    mse = jnp.mean(optax.l2_loss(x, x_pred))
    log_lkhood = mse / (2.0 * pl_sig**2)

    return key, log_lkhood