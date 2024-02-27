import flax.linen as nn

class GEN(nn.Module):
    input_dim: int
    feature_dim: int
    output_dim: int
    image_dim: int
    leak_coef: float

    @nn.compact
    def __call__(self, z):
        f = nn.activation.leaky_relu
        z = z.squeeze()
        z = nn.ConvTranspose(self.input_dim, self.feature_dim * 16, kernel_size=(4, 4), strides=(1, 1), padding='VALID')(z)
        z = nn.BatchNorm()(z)
        z = self.f(z, self.leak_coef)
        z = nn.ConvTranspose(self.feature_dim * 16, self.feature_dim * 8, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.BatchNorm()(z)
        z = self.f(z, self.leak_coef)
        z = nn.ConvTranspose(self.feature_dim * 8, self.feature_dim * 4, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
        z = nn.BatchNorm()(z)
        z = self.f(z, self.leak_coef)

        if self.image_dim == 64: # 64 x 64 images, i.e CelebA
            z = nn.ConvTranspose(self.feature_dim * 4, self.feature_dim * 2, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
            z = self.f(z, self.leak_coef)
            z = nn.ConvTranspose(self.feature_dim * 2, self.output_dim, kernel_size=(4, 4), strides=(2, 2), padding='SAME')(z)
            z = nn.activation.tanh(z)
        else: # 32 x 32 images, i.e. CIFAR10, SVHN
            z = nn.ConvTranspose(self.feature_dim * 4, self.output_dim, kernel_size=(4, 4), strides=(1, 1), padding='SAME')(z)
            z = nn.activation.tanh(z)

        return z