import flax.linen as nn

class EBM(nn.Module):
    hidden_units: int
    output_dim: int
    leak_coef: float
    f: nn.Module = nn.leaky_relu(leak_coef)

    @nn.compact
    def __call__(self, z):
        z = z.squeeze()
        z = nn.Dense(self.hidden_units)(z)
        z = self.f(z)
        z = nn.Dense(self.hidden_units)(z)
        z = self.f(z)
        z = nn.Dense(self.output_dim)(z)
        return z.view(-1, self.output_dim, 1, 1)