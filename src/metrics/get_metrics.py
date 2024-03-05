import jax

from src.metrics.biased_metrics.frechet_inception_distance import calculate_fid, calculate_mifid
from src.metrics.biased_metrics.kernel_inception_distance import calculate_kid

@jax.jit
def get_metrics(x_features, x_pred_features):

    # Calculate the FID
    fid = calculate_fid(x_features, x_pred_features)

    # Calculate the MIFID
    mifid = calculate_mifid(x_features, x_pred_features)

    # Calculate the KID
    kid = calculate_kid(x_features, x_pred_features)

    return fid, mifid, kid
