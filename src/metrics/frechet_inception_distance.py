import jax.numpy as jnp

def calculate_fid(real_features, fake_features):
    """
    Frechet Inception Distance.
    
    https://doi.org/10.48550/arXiv.1706.08500
    """

    # Mean and covariances
    mu_real, std_real = real_features.mean(axis=0), jnp.cov(real_features, rowvar=False)
    mu_gen, std_gen = fake_features.mean(axis=0), jnp.cov(fake_features, rowvar=False)

    # Calculate sum of squared differences
    ssd = jnp.sum((mu_real - mu_gen) ** 2)

    # Calculate square root of product of covariances
    cov_mean = jnp.sqrt(std_real @ std_gen)

    # Check and correct imaginary numbers
    if jnp.iscomplexobj(cov_mean):
        cov_mean = cov_mean.real

    # Calculate FID
    fid = ssd + jnp.trace(std_real + std_gen - 2 * cov_mean)

    return fid

def cosine_similarity(real_features, fake_features):
         
        # Calculate the norm
        real_norm = jnp.linalg.norm(real_features, axis=1, keepdims=True)
        fake_norm = jnp.linalg.norm(fake_features, axis=1, keepdims=True)

        # Calculate cosine similarity
        cosine_sim = jnp.dot(real_features, fake_features.T) / (real_norm * fake_norm.T)

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
