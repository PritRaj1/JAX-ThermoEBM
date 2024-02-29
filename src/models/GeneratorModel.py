import flax.linen as nn
import configparser

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

feature_dim = int(parser['GEN']['GEN_FEATURE_DIM'])
output_dim = int(parser['GEN']['CHANNELS'])
leak_coef = float(parser['GEN']['GEN_LEAK'])

class GEN(nn.Module):
    image_dim: int

    def setup(self):

        self.f = nn.activation.leaky_relu

        def conditional_64(z):
            z = nn.ConvTranspose(
                feature_dim * 2, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
            )(z)
            z = self.f(z, negative_slope=leak_coef)
            return z

        def conditional_32(z):
            return z

        if self.image_dim == 64:
            self.conditional_block = conditional_64

        else:
            self.conditional_block = conditional_32

    @nn.compact
    def __call__(self, z):

        z = nn.ConvTranspose(
            feature_dim * 16, (4, 4), (4, 4), padding="CIRCULAR", use_bias=False
        )(z)
        z = self.f(z, negative_slope=leak_coef)
        z = nn.ConvTranspose(
            feature_dim * 8, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = self.f(z, negative_slope=leak_coef)
        z = nn.ConvTranspose(
            feature_dim * 4, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = self.f(z, negative_slope=leak_coef)
        z = self.conditional_block(z)
        z = nn.ConvTranspose(
            output_dim, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = nn.tanh(z)

        return z 