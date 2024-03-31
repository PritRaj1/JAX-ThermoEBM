import jax
import optax
import configparser

from src.MCMC_Samplers.sample_distributions import sample_prior, sample_posterior
from src.MCMC_Samplers.grad_log_probs import log_llood_fcn, log_prior_fcn

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

p0_sig = float(parser["SIGMAS"]["p0_SIGMA"])
pl_sig = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])


def get_ebm_energies(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns  f(z|x, t=1) - log(π(z)), f(z), z ~ p(z|x, t=1)"""
    key, z_prior = sample_prior(key, EBM_params, EBM_fwd)
    z_posterior = sample_posterior(key, x, 1, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
    en_pos = log_prior_fcn(z_posterior, EBM_params, EBM_fwd)
    en_neg = log_prior_fcn(z_prior, EBM_params, EBM_fwd)

    return en_pos, en_neg, z_posterior


batch_energies = jax.vmap(get_ebm_energies, in_axes=(0, 0, None, None, None, None))


def mean_EBMloss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns E_{z|x}[ f(z) ] - E_{z}[ f(z) ]"""
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    en_pos, en_neg, z_posterior = batch_energies(
        subkey_batch, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    return (en_pos.mean() - en_neg.mean()), z_posterior


batch_llhood = jax.vmap(log_llood_fcn, in_axes=(0, 0, None, None, None))


def mean_GENloss(x, z_posterior, GEN_params, GEN_fwd):
    """Returns E_{z|x,t}[ log[ p_β(x | z, t=1) ] ]"""
    llhood = batch_llhood(z_posterior, x, 1, GEN_params, GEN_fwd)
    return llhood.mean()
