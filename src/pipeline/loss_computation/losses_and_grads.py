import jax
import configparser
from jax import value_and_grad
from jax.lax import stop_gradient

from src.pipeline.update_steps import *
from src.pipeline.loss_computation.vanilla_computation import vanilla_loss
from src.pipeline.loss_computation.thermo_computation import thermo_loss

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")
temp_power = float(parser["TEMP"]["TEMP_POWER"])

if temp_power > 0:
    print("Using Temperature Schedule with Power: {}".format(temp_power))
    loss_fcn = thermo_loss
else:
    loss_fcn = vanilla_loss
    print("Using no Thermodynamic Integration, defaulting to Vanilla Model")

def get_losses_and_grads(key, x, params_tup, fwd_fcn_tup):
    """Function to compute the losses and gradients of the thermodynamic models."""

    ebm_params, gen_params = params_tup

    # Compute loss of both models
    total_loss, (grad_ebm, grad_gen) = value_and_grad(loss_fcn, argnums=(2, 3))(
        key, stop_gradient(x), ebm_params, gen_params, *fwd_fcn_tup
    )

    return total_loss, grad_ebm, grad_gen