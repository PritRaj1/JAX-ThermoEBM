import jax
import numpy as np
import configparser
import optax

from src.MCMC_Samplers.sample_distributions import sample_p0
from src.models.PriorModel import EBM
from src.models.GeneratorModel import GEN

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

E_lr = float(parser["OPTIMIZER"]["E_LR"])
G_lr = float(parser["OPTIMIZER"]["G_LR"])
E_gamma = float(parser["OPTIMIZER"]["E_GAMMA"])
G_gamma = float(parser["OPTIMIZER"]["G_GAMMA"])
E_opt_steps = int(parser["OPTIMIZER"]["E_STEPS"])
G_opt_steps = int(parser["OPTIMIZER"]["G_STEPS"])
temp_power = float(parser["TEMP"]["TEMP_POWER"])
num_temps = int(parser["TEMP"]["NUM_TEMPS"])


def init_EBM(key):
    key, z_init = sample_p0(key)

    EBM_model = EBM()

    EBM_params = EBM_model.init(key, z_init)

    EBM_fwd = jax.vmap(EBM_model.apply, in_axes=(None, 0))

    return EBM_params, EBM_fwd


def init_GEN(key, image_dim):
    key, z_init = sample_p0(key)

    GEN_model = GEN(image_dim)
    GEN_params = GEN_model.init(key, z_init)

    GEN_fwd = jax.vmap(GEN_model.apply, in_axes=(None, 0))

    return GEN_params, GEN_fwd


def init_GEN_optimiser(GEN_params):
    GEN_schedule = optax.exponential_decay(G_lr, G_gamma, G_opt_steps)
    GEN_optimiser = optax.adam(GEN_schedule)
    GEN_opt_state = GEN_optimiser.init(GEN_params)

    return GEN_optimiser, GEN_opt_state


def init_EBM_optimiser(EBM_params):
    E_schedule = optax.exponential_decay(E_lr, E_gamma, E_opt_steps)
    E_optimiser = optax.adam(E_schedule)
    E_opt_state = E_optimiser.init(EBM_params)

    return E_optimiser, E_opt_state


def init_temp_schedule():
    if temp_power >= 1:
        print("Using temperature schedule with power: {}".format(temp_power))
        temp = tuple(np.linspace(0, 1, num_temps) ** temp_power)

    else:
        print("Using no thermodynamic integration, defaulting to Vanilla Model")
        temp = (1,)

    return temp
