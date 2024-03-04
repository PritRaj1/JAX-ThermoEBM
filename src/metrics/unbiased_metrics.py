
import jax
import jax.numpy as jnp
import numpy as np
from sklearn.linear_model import LinearRegression

from src.metrics.get_metrics import get_metrics
from src.metrics.inception_network import extract_features

def profile_generation(
    key, x, gen_fcn, min_samples=100, max_samples=2000, num_points=50, num_plot=4
):
    """
    Function to profile the generative capacity of the model.
    Traditional image-quality metrics are generally biased,
    so following the previous work of Chong et. al, we aim to
    remove bias by interpolating to an infinite sample size.

    https://doi.org/10.48550/arXiv.1911.07023

    Args:
    - key: PRNG key
    - x: all test data
    - gen_fcn: generative function, partially preloaded with immutables
    - min_samples: minimum number of samples
    - max_samples: maximum number of samples
    - num_points: number of points to profile
    - num_plot: number of images to plot for visual inspection

    Returns:
    - fid_inf: the unbiased FID
    - mifid_inf: the unbiased MIFID
    - kid_inf: the unbiased KID
    - four_real: 4 random real images
    - four_fake: 4 random fake images
    """

    batch_sizes = jnp.linspace(min_samples, max_samples, num_points, dtype=int)
    
    # Generate the maximum number of samples
    key, x_pred = gen_fcn(key, num_images=max_samples)

    # Get all activations
    x_activations = extract_features(x)
    x_pred_activations = extract_features(x_pred)

    fid = np.zeros(num_points)
    mifid = np.zeros(num_points)
    kid = np.zeros(num_points)

    for idx, sample_size in enumerate(batch_sizes):
        
        # Randomly sample from the activations
        x_act_i = np.random.shuffle(x_activations)[:sample_size]
        x_pred_act_i = np.random.shuffle(x_pred_activations)[:sample_size]

        # Compute metrics for this sample size
        fid[idx] , mifid[idx], kid[idx] = get_metrics(x_act_i, x_pred_act_i)
    
    # Fit a linear regression to the inverse of the batch sizes
    reg_fid = LinearRegression().fit(1/batch_sizes.reshape(-1, 1), fid)
    reg_mifid = LinearRegression().fit(1/batch_sizes.reshape(-1, 1), mifid)
    reg_kid = LinearRegression().fit(1/batch_sizes.reshape(-1, 1), kid)

    # Interpolate to infinite sample size
    fid_inf = reg_fid.predict(np.array([[0]]))[0,0]
    mifid_inf = reg_mifid.predict(np.array([[0]]))[0,0]
    kid_inf = reg_kid.predict(np.array([[0]]))[0,0]

    # Return 4 random images for plotting (visual inspection)
    random_indices = np.random.choice(len(x_pred), num_plot, replace=False)
    four_real = np.array([x[i] for i in random_indices])
    four_fake = np.array([x_pred[i] for i in random_indices])

    return key, fid_inf, mifid_inf, kid_inf, four_real, four_fake


        



