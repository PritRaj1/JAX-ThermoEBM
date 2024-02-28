import flax.linen as nn
import jax.numpy as jnp


class GEN(nn.Module):
    feature_dim: int
    output_dim: int
    image_dim: int
    leak_coef: float

    def setup(self):

        self.f = nn.activation.leaky_relu

        def conditional_64(z):
            z = nn.ConvTranspose(
                self.feature_dim * 2, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
            )(z)
            z = self.f(z, negative_slope=self.leak_coef)
            return z

        def conditional_32(z):
            return z

        if self.image_dim == 64:
            self.conditional_block = conditional_64

        else:
            self.conditional_block = conditional_32

    @nn.compact
    def __call__(self, z):

        z = jnp.transpose(z, (0, 3, 2, 1))
        z = nn.ConvTranspose(
            self.feature_dim * 16, (4, 4), (4, 4), padding="CIRCULAR", use_bias=False
        )(z)
        z = self.f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 8, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = self.f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 4, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = self.f(z, negative_slope=self.leak_coef)
        z = self.conditional_block(z)
        z = nn.ConvTranspose(
            self.output_dim, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = nn.tanh(z)

        return jnp.transpose(z, (0, 3, 2, 1))


class GEN_64(nn.Module):
    feature_dim: int
    output_dim: int
    image_dim: int
    leak_coef: float

    @nn.compact
    def __call__(self, z):

        f = nn.activation.leaky_relu

        z = z.reshape((z.shape[0], 1, 1, -1))
        z = nn.ConvTranspose(
            self.feature_dim * 16, (4, 4), (4, 4), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 8, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 4, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 2, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.output_dim, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = nn.tanh(z)

        return z.reshape((z.shape[0], self.output_dim, self.image_dim, self.image_dim))


class GEN_32(nn.Module):
    feature_dim: int
    output_dim: int
    image_dim: int
    leak_coef: float

    @nn.compact
    def __call__(self, z):

        f = nn.activation.leaky_relu

        z = z.reshape((z.shape[0], 1, 1, -1))
        z = nn.ConvTranspose(
            self.feature_dim * 16, (4, 4), (4, 4), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 8, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.feature_dim * 4, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(
            self.output_dim, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False
        )(z)
        z = nn.tanh(z)

        return z.reshape((z.shape[0], self.output_dim, self.image_dim, self.image_dim))
