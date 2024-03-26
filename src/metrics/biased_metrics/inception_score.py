import jax.numpy as jnp
from optax import kl_divergence as kl_div
from flax.linen import softmax, log_softmax
import jax

def scan_kl(carry, x):
    """Scan function for KL divergence"""
    return None, jnp.sum(kl_div(log_softmax(x), jnp.mean(x, axis=0)))

@jax.jit
def calculate_is(fake_features, splits=10):
    """exp(E_{x ~ p(x)}[KL(p(y|x) || p(y))])"""

    prob = softmax(fake_features, axis=-1)
    split_indices = [i * (prob.shape[0] // splits) for i in range(splits + 1)]
    parts = jnp.array_split(prob, split_indices)
    _, kl_div_stack = jax.lax.scan(scan_kl, None, jnp.array(parts[1:-1]))

    return jnp.exp(jnp.mean(kl_div_stack, axis=0))