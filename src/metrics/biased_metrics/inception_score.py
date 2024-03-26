import jax.numpy as jnp
from optax import kl_divergence as kl_div
from flax.linen import softmax, log_softmax
import jax

def scan_kl(carry, x):
    """Scan function for KL divergence"""
    return None, kl_div(log_softmax(x), jax.lax.expand_dims(jnp.mean(x, axis=0), 0))

def calculate_is(fake_features, splits=10):
    """exp(E_{x ~ p(x)}[KL(p(y|x) || p(y))])"""

    prob = softmax(fake_features)

    parts = jnp.split(prob, splits)
    _, kl_div_stack = jax.lax.scan(scan_kl, None, parts)

    return jnp.exp(jnp.sum(kl_div_stack, axis=0) / splits)