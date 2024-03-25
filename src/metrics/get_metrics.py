import jax

from src.metrics.biased_metrics.frechet_inception_distance import calculate_fid, calculate_mifid
from src.metrics.biased_metrics.kernel_inception_distance import calculate_kid
from src.metrics.biased_metrics.inception_score import calculate_is

def get_metrics(train_x_features, val_x_features, x_pred_features):

    fid = calculate_fid(val_x_features, x_pred_features)
    mifid = calculate_mifid(train_x_features, val_x_features, x_pred_features)
    kid = calculate_kid(val_x_features, x_pred_features)
    incep = calculate_is(x_pred_features)

    return fid, mifid, kid, incep
