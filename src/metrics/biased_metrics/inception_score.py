import jax.numpy as jnp
from optax import kl_divergence as kl_div
from flax.linen import softmax, log_softmax

def calculate_is(fake_features):
    """exp(E_{x ~ p(x)}[KL(p(y|x) || p(y))])"""

    prob = softmax(fake_features)
    marginal = prob.mean(axis=0)
    log_conditional = log_softmax(fake_features)
    kl_divergence = kl_div(log_conditional, marginal)

    return jnp.exp(kl_divergence.mean())