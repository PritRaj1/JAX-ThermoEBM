import flax.linen as nn
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

hidden_units = int(parser['EBM']['EBM_FEATURE_DIM'])
output_dim = int(parser['EBM']['Z_CHANNELS'])
activation_coef = float(parser['EBM']['EBM_ACTIVATION_COEF'])

class EBM(nn.Module):

    @nn.compact
    def __call__(self, z):
        f = nn.activation.leaky_relu
        
        z = nn.Dense(hidden_units)(z)
        z = f(z, activation_coef)
        z = nn.Dense(hidden_units)(z)
        z = f(z, activation_coef)
        z = nn.Dense(output_dim)(z)
        z = f(z, activation_coef)
        
        return z
