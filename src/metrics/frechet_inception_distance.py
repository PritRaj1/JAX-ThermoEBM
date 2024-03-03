import jax.numpy as jnp


def calculate_fid(real_features, fake_features):
    """
    Frechet Inception Distance.

    https://doi.org/10.48550/arXiv.1706.08500
    """

    # Mean and covariances
    mu_real = real_features.mean(axis=-1)
    mu_fake = fake_features.mean(axis=-1)

    var_real = jnp.cov(real_features, real_features, rowvar=True)
    var_fake = jnp.cov(fake_features, fake_features, rowvar=True)

    diff = mu_real - mu_fake
    
    return  jnp.dot(diff, diff) + jnp.trace(var_real + var_fake - 2 * jnp.sqrt(var_real @ var_fake))


def cosine_similarity(real_features, fake_features):

    # Calculate the norm
    real_norm = jnp.linalg.norm(real_features, axis=1, keepdims=True)
    fake_norm = jnp.linalg.norm(fake_features, axis=1, keepdims=True)

    # Calculate cosine similarity
    cosine_sim = jnp.dot(real_features, fake_features.T) / (real_norm * fake_norm.T)

    return cosine_sim.sum() / (fake_features.shape[0])


def calculate_mifid(real_features, fake_features):
    """
    Memorisation-information FID.

    https://doi.org/10.48550/arXiv.1911.07023
    """

    # Calculate FID
    fid = calculate_fid(real_features, fake_features)

    # Calculate cosine similarity
    cosine_sim = cosine_similarity(real_features, fake_features)

    # Calculate MIFID
    mifid = fid * (1 - cosine_sim)

    return mifid
