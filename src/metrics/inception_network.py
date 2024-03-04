from tensorflow.keras.applications import InceptionV3
import numpy as np
import configparser
from jax.image import resize
import jax

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

inception_model = InceptionV3(weights="imagenet", include_top=True, pooling="avg", input_shape=(75, 75, 3))


def preprocess_image(image):
    image = resize(image, (75, 75, 3), method="bilinear")
    return image


preprocess = jax.vmap(preprocess_image, in_axes=(0))


# Define the function to extract features
def extract_features(images):
    images = np.asanyarray(preprocess(images))
    features = inception_model(images)
    return features
