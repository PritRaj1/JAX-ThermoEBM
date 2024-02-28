import flax.linen as nn

class GEN(nn.Module):
    feature_dim: int
    output_dim: int
    image_dim: int
    leak_coef: float

    @nn.compact
    def __call__(self, z):


        f = nn.activation.leaky_relu

        z = z.reshape((z.shape[0], 1, 1, -1))
        z = nn.ConvTranspose(self.feature_dim * 16, (4, 4), (4, 4), padding="CIRCULAR", use_bias=False)(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(self.feature_dim * 8, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False)(z)
        z = f(z, negative_slope=self.leak_coef)
        z = nn.ConvTranspose(self.feature_dim * 4, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False)(z)
        z = f(z, negative_slope=self.leak_coef)
        
        if self.image_dim == 64:
            z = nn.ConvTranspose(self.feature_dim * 2, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False)(z)
            z = f(z, negative_slope=self.leak_coef)

        z = nn.ConvTranspose(self.output_dim, (4, 4), (2, 2), padding="CIRCULAR", use_bias=False)(z)
        z = nn.tanh(z)

        return z