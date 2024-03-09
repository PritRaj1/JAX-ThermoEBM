from tensorflow.keras.applications import InceptionV3
import jax
from jax.image import resize
from functools import partial
import numpy as np
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

inception_model = InceptionV3(
    weights="imagenet", include_top=False, pooling="avg", input_shape=(75, 75, 3)
)


def inception_pass_callback(x):
    x = np.expand_dims(x, axis=0)
    return inception_model(x)[0]

def fwd_pass(carry, images, result_shape):
    return carry, jax.pure_callback(inception_pass_callback, result_shape, images, vectorized=False)

def extract_features(images):

    images = resize(images, (images.shape[0], 75, 75, 3), method="bilinear")
    scan_images = partial(fwd_pass, result_shape=jax.core.ShapedArray((2048,), images.dtype)) 
    _, features = jax.lax.scan(f=scan_images, init=None, xs=images)

    return features
