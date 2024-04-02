import jax
import jax.numpy as jnp
from jax.lax import stop_gradient
import configparser

from src.MCMC_Samplers.sample_distributions import sample_posterior, sample_prior
from src.MCMC_Samplers.log_pdfs import log_llood_fcn, log_prior_fcn
from src.MCMC_Samplers.grad_log_probs import sum_llhood

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
pl_sigma = float(parser["SIGMAS"]["LKHOOD_SIGMA"])
image_dim = 64 if parser["PIPELINE"]["DATASET"] == "CelebA" else 32
m = 3 * image_dim * image_dim

batched_posterior = jax.vmap(
    sample_posterior, in_axes=(0, 0, None, None, None, None, None)
)


def batch_sample_posterior(key, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns a batch of samples from the posterior distribution at a given temperature."""

    # Sample batch amount of z_posterior samples
    key_batch = jax.random.split(key, batch_size + 1)
    key, subkey_batch = key_batch[0], key_batch[1:]
    z_posterior = batched_posterior(
        subkey_batch, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd
    )

    return key, z_posterior


def prior_norm(key, EBM_params, EBM_fwd):
    """Returns the normalisation constant for the prior distribution."""
    key, z = sample_prior(key, EBM_params, EBM_fwd)
    return log_prior_fcn(z, EBM_params, EBM_fwd).sum()


get_priors = jax.vmap(prior_norm, in_axes=(0, None, None))


def marginal_llhood(x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """
    Returns log[ p_β(x|z,t) ] + log[ p_α(z) ] - log[ Z_{θ,t} ]

    If t=1 AND z ~ p(z|x,t=1), this is the value of the marginal logpdf.
    If t>1, this is the value of the posterior logpdf at temperature t for a given z.
    The normalisation is very costly, but we cannot call stop_gradient on it, since
    Z_{θ,t} is parameter dependent. Without normalisation, it becomes difficult
    to compare the marginal logpdfs between the TI and Vanilla losses.

    Likelihood normalisation is not paramter dependent, but is dependent on t, so 
    we apply logsumexp with stop grad. Prior normalisation is conducted outside of this function.
    """

    llhood = log_llood_fcn(z, x, t, GEN_params, GEN_fwd)
    prior = log_prior_fcn(z, EBM_params, EBM_fwd)
    llhood -= jax.scipy.special.logsumexp(stop_gradient(llhood))

    return llhood.sum() + prior.sum()


marginal = jax.vmap(marginal_llhood, in_axes=(0, 0, None, None, None, None, None))
batched_sum_llhood = jax.vmap(sum_llhood, in_axes=(0, 0, None, None, None))


def batched_marginal_llhood(key, x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    logpdf = marginal(x, z, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd)

    # Prior normalisation is parameter dependent = E_{z~p_α(z)}[ exp(f_α(z)) ], which is easier to track grads for than logsum.
    keybatch = jax.random.split(key, batch_size+1)
    key, subkey_batch = keybatch[0], keybatch[1:]
    logpdf -= get_priors(subkey_batch, EBM_params, EBM_fwd).mean()
    return logpdf


def mean_llhood(x, z, GEN_params, GEN_fwd):
    """Returns the expected normalised log-likelihood of the generator model."""

    # Untempered, therefore exact normalisation constant is feasible.
    llhood = batched_sum_llhood(z, x, 1, GEN_params, GEN_fwd)
    llhood -= 0.5 * m * jnp.log(2 * jnp.pi * pl_sigma**2)
    return llhood.mean()
