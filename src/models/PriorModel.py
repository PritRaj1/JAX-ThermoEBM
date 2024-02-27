import flax.linen as nn

class EBM(nn.Module):
    hidden_units: int
    output_dim: int
    leak_coef: float

    @nn.compact
    def __call__(self, z):
        f = nn.activation.leaky_relu
        z = z.squeeze()
        z = nn.Dense(self.hidden_units)(z)
        z = f(z, self.leak_coef)
        z = nn.Dense(self.hidden_units)(z)
        z = f(z, self.leak_coef)
        z = nn.Dense(self.output_dim)(z)

        return z.reshape((-1, self.output_dim, 1, 1))