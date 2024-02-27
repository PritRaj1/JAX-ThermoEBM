import jax

def sample_z(state, data):

    # Parse samplers
    prior_sampler = state.samplers['prior']
    posterior_sampler = state.samplers['posterior']

    # Initialise noisy sample
    key, subkey = jax.random.split(state.key)
    z0 = prior_sampler.sample_p0(subkey) # z0 ~ p_0(z)
    state.key = key

    # MCMC sampling to generate zK
    zK_prior = prior_sampler(z0, state, key) # zK ~ p_a(z)
    zK_posterior = posterior_sampler(z0, state, key, data) # zK ~ p_θ(z|x)

    return zK_prior, zK_posterior

def generate(state):

    prior_sampler = state.samplers['prior']

    key, subkey = jax.random.split(state.key)
    z0 = prior_sampler.sample_p0(subkey)
    state.key = key

    z_prior = prior_sampler(z0, state, key)

    x_pred = state.model_apply['GEN_apply'](state.params['GEN_params'], jax.lax.stop_gradient(z_prior))

    return x_pred