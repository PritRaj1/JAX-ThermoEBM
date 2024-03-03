import jax
import jax.numpy as jnp
import configparser
import optax

from src.MCMC_Samplers.sample_distributions import sample_p0
from src.models.PriorModel import EBM
from src.models.GeneratorModel import GEN

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

E_lr = float(parser["OPTIMIZER"]["E_LR"])
G_lr = float(parser["OPTIMIZER"]["G_LR"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])


def init_EBM(key):
    key, z_init = sample_p0(key)

    EBM_model = EBM()

    EBM_params = EBM_model.init(key, z_init)

    GEN_fwd = EBM_model.apply

    return key, EBM_params, GEN_fwd


def init_GEN(key, image_dim):
    key, z_init = sample_p0(key)

    GEN_model = GEN(image_dim)
    GEN_params = GEN_model.init(key, z_init)

    GEN_fwd = GEN_model.apply

    return key, GEN_params, GEN_fwd


def init_GEN_optimiser(GEN_params):
    GEN_optimiser = optax.adam(G_lr)
    GEN_opt_state = GEN_optimiser.init(GEN_params)

    return GEN_optimiser, GEN_opt_state


def init_EBM_optimiser(EBM_params):
    E_optimiser = optax.adam(E_lr)
    E_opt_state = E_optimiser.init(EBM_params)

    return E_optimiser, E_opt_state


def init_temp_schedule():
    if temp_power >= 1:
        print("Using Temperature Schedule with Power: {}".format(temp_power))
        temp = jnp.linspace(0, 1, num_temps) ** temp_power
        print("Temperature Schedule: {}".format(temp))

    else:
        print("Using no Thermodynamic Integration, defaulting to Vanilla Model")
        temp = jnp.array([1])
        print("Temperature Schedule: {}".format(temp))

    return temp
