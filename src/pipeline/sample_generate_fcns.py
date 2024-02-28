import jax

def sample_z(state, z0, data, t):

    # Parse samplers
    prior_sampler = state.samplers['prior']
    posterior_sampler = state.samplers['posterior']

    # Initialise noisy sample
    key, subkey = jax.random.split(state.key)

    # MCMC sampling to generate zK
    zK_prior = prior_sampler(z0, state, subkey) # zK ~ p_a(z)
    zK_posterior = posterior_sampler(z0, state, subkey, data, t) # zK ~ p_θ(z|x, t)

    return zK_prior, zK_posterior

def generate(state):

    prior_sampler = state.samplers['prior']

    key, subkey = jax.random.split(state.key)
    z0 = prior_sampler.sample_p0(subkey)
    state.key = key

    z_prior = prior_sampler(z0, state, key)

    x_pred = state.model_apply['GEN_apply'](state.params['GEN_params'], jax.lax.stop_gradient(z_prior))

    return x_pred