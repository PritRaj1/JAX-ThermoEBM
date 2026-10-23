import jax
import jax.numpy as jnp
from blackjax.smc import resampling
from flax import nnx

resampler_map = {
	"residual": resampling.residual,
	"systematic": resampling.systematic,
	"stratified": resampling.stratified,
	"multinomial": resampling.multinomial,
}


def pairwise_sqdist(xi: jax.Array, xj: jax.Array) -> jax.Array:
	return jnp.sum((xi - xj) ** 2)


class ImportanceTuner(nnx.Module):
	def __init__(self, resampler_type: str):
		self.resampler = resampler_map.get(resampler_type, resampling.residual)

	def logllhood(self, model: nnx.Module, z: jax.Array, x: jax.Array) -> jax.Array:
		x_pred = model.gen(z)
		se = jnp.sum(
			(x[:, None, ...] - x_pred[None, :, ...]) ** 2,
			axis=tuple(range(2, x.ndim + 1)),
		)
		return -se / (2 * model.gen.sigma**2)

	@nnx.jit
	def batch_resample(
		self, subkeys: jax.Array, model: nnx.Module, z: jax.Array, x: jax.Array
	) -> tuple[jax.Array, jax.Array, jax.Array]:
		def resample_one(key, weights):
			return self.resampler(key, weights, z.shape[0])

		ll = self.logllhood(model, z, x)
		weights = jax.nn.softmax(ll, axis=-1)
		idx = jax.vmap(resample_one)(subkeys, weights)
		weights = jnp.take_along_axis(weights, idx, axis=1)
		return idx, jax.nn.softmax(weights, axis=-1)

	def __call__(
		self,
		model: jax.Array,
		weights: jax.Array,
		x: jax.Array,
		z_post: jax.Array,
		z_prior: jax.Array,
	) -> jax.Array:
		N = x.shape[0]

		# IS weighed expected log-likelihood
		x_gen = jax.vmap(model.gen)(z_post)
		se = jnp.sum(
			((jnp.expand_dims(x, axis=1) - x_gen) ** 2),
			axis=tuple(range(2, x.ndim + 1)),
		)
		ll = (weights * se).mean()

		# IS weighed expected log-prior
		def posterior_energy(z: jax.Array) -> jax.Array:
			return jax.vmap(model.ebm.en)(z)

		e_post = (weights * jax.vmap(posterior_energy)(z_post)).mean()
		e_prior = model.ebm.en(z_prior) / N

		contrastive_div = -e_post + e_prior
		return contrastive_div + ll
