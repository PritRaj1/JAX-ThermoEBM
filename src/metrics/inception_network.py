from tensorflow.image import resize
from tensorflow.keras.applications import InceptionV3
import numpy as np
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

inception_model = InceptionV3(weights="imagenet", include_top=False, pooling="avg", input_shape=(75, 75, 3))


def extract_features(images):
    images = np.asarray(images)
    images = resize(images, (75, 75))
    features = np.zeros((images.shape[0], 2048))

    for idx, image in enumerate(images):
        image = np.expand_dims(image, axis=0)
        features[idx] = np.asarray(inception_model(image))

    return features 
