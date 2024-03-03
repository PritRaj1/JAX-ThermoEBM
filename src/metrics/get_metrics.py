import numpy as np
from tensorflow.keras.applications import InceptionV3
import configparser

from src.metrics.frechet_inception_distance import calculate_fid, calculate_mifid
from src.metrics.kernel_inception_distance import calculate_kid

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

img_dim = 64 if parser["PIPELINE"]["DATASET"] == "CIFAR10" else 32

inception_model = InceptionV3(weights='imagenet', 
                              include_top=False, 
                              pooling='avg', 
                              input_shape=(img_dim, img_dim, 3))

def metrics(x, x_pred):

    # Convert for [-1, 1] to [0, 1], for inception model
    x = (x + 1) / 2
    x_pred = (x_pred + 1) / 2

    # Get the inception features
    x_features = inception_model.predict(np.asarray(x))
    x_pred_features = inception_model.predict(np.asarray(x_pred))

    # Calculate the FID
    fid = calculate_fid(x_features, x_pred_features)

    # Calculate the MIFID
    mifid = calculate_mifid(x_features, x_pred_features)

    # Calculate the KID
    kid = calculate_kid(x_features, x_pred_features)

    return fid, mifid, kid



