import jax

from src.metrics.biased_metrics.frechet_inception_distance import calculate_fid, calculate_mifid
from src.metrics.biased_metrics.kernel_inception_distance import calculate_kid

def get_metrics(x_features, x_pred_features):

    fid = calculate_fid(x_features, x_pred_features)
    mifid = calculate_mifid(x_features, x_pred_features)
    kid = calculate_kid(x_features, x_pred_features)

    return fid, mifid, kid
