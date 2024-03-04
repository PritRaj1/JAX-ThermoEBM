from src.metrics.frechet_inception_distance import calculate_fid, calculate_mifid
from src.metrics.kernel_inception_distance import calculate_kid
from src.metrics.inception_network import extract_features


def profile_image(x, x_pred):

    # Get the inception features
    x_features = extract_features(x)
    x_pred_features = extract_features(x_pred)

    # Calculate the FID
    fid = calculate_fid(x_features, x_pred_features)

    # Calculate the MIFID
    mifid = calculate_mifid(x_features, x_pred_features)

    # Calculate the KID
    kid = calculate_kid(x_features, x_pred_features)

    return fid, mifid, kid
