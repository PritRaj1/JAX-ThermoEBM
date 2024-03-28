import jax
import optax
import configparser

from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])


def get_ebm_energies(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns  f(z|x, t=1), f(z), z ~ p(z|x, t=1)"""
    key, z_prior = sample_prior(key, EBM_params, EBM_fwd)
    z_posterior = sample_posterior(key, x, 1, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
    return EBM_fwd(EBM_params, z_posterior), EBM_fwd(EBM_params, z_prior), z_posterior


batch_energies = jax.vmap(get_ebm_energies, in_axes=(0, 0, None, None, None, None))


def mean_EBMloss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns E_{z|x}[ f(z) ] - E_{z}[ f(z) ]"""
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    en_pos, en_neg, z_posterior = batch_energies(
        subkey_batch, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    return (en_pos - en_neg).mean(), z_posterior


def llhood(x, z, GEN_params, GEN_fwd):
    """Returns log[ p_β(x | z, t)"""

    # Generate a sample from the generator
    x_pred = GEN_fwd(GEN_params, z)

    # Compute log[ p_β(x | z) ] = 1/2 * (x - g(z))^2 / σ^2
    mse = optax.l2_loss(x, x_pred).sum()
    log_lkhood = -mse / (2.0 * pl_sig**2)

    return log_lkhood


batch_llhood = jax.vmap(llhood, in_axes=(0, 0, None, None))


def mean_GENloss(x, z_posterior, GEN_params, GEN_fwd):
    """Returns E_{z|x,t}[ log[ p_β(x | z, t) ] ]"""
    loss = batch_llhood(x, z_posterior, GEN_params, GEN_fwd)
    return loss.mean()
