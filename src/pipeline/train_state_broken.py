import jax
import jax.numpy as jnp
import optax
# from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.pipeline.sample_generate_fcns import sample_z
from src.pipeline.loss_fcn import discretised_TI_loss_fcn, ebm_loss, gen_loss

loss_fcn_EBM = jax.jit(discretised_TI_loss_fcn(ebm_loss, gen=False))
loss_fcn_GEN = jax.jit(discretised_TI_loss_fcn(gen_loss, gen=True))

@jax.jit
def get_losses(x, state):

    loss_ebm = loss_fcn_EBM(state, x)
    loss_gen = loss_fcn_GEN(state, x)
    
    grad_ebm = jax.grad(loss_fcn_EBM)(state.params["EBM_params"], x)    
    grad_gen = jax.grad(loss_fcn_GEN)(state.params["GEN_params"], x)

    return loss_ebm, loss_gen, grad_ebm, grad_gen

class Train_State():
    def __init__(self, 
                 key,
                 EBM_model, 
                 GEN_model, 
                 EBM_optimiser, 
                 GEN_optimiser,
                 prior_sampler,
                 posterior_sampler,
                 temperature_power=0,
                 num_temps=1,
                 log_path=None):
        
        key, subkey = jax.random.split(key)
        z_init = prior_sampler.sample_p0(subkey)

        self.key = key

        init_rng = jax.random.PRNGKey(0)
        self.params = {
            "EBM_params": EBM_model.init(init_rng, z_init),   
            "GEN_params": GEN_model.init(init_rng, z_init)
        }
        del init_rng 
        
        self.model_apply = {
            "EBM_apply": EBM_model.apply,
            "GEN_apply": GEN_model.apply,
        }

        self.optimisers = {
            "EBM_opt": EBM_optimiser,
            "GEN_opt": GEN_optimiser
        }

        self.opt_states = {
            "EBM_opt_state": self.optimisers["EBM_opt"].init(self.params["EBM_params"]),
            "GEN_opt_state": self.optimisers["GEN_opt"].init(self.params["GEN_params"])
        }

        prior_sampler.config(self.model_apply["EBM_apply"])
        posterior_sampler.config(self.model_apply["EBM_apply"], self.model_apply["GEN_apply"])

        self.samplers = {
            "prior": prior_sampler,
            "posterior": posterior_sampler
        }

        if temperature_power >= 1:
            print("Using temperature schedule with power: {}".format(temperature_power))
            self.temp = {
                "schedule": jnp.linspace(0, 1, num_temps)**temperature_power,
            }

        else:
            print("Using no thermodynamic integration, defaulting to Vanilla Model")
            self.temp = {
                "schedule": jnp.array([1]),
            }

        self.loggers = {
            "tb_writer": None,
            "csv_logger": None
        }

    def training_step(self, x):

        ebm_loss, gen_loss, grad_ebm, grad_gen = get_losses(x, self)

        self.params["EBM_params"], self.opt_states["EBM_opt_state"] = self.update_params(grad_ebm, self.opt_states["EBM_opt_state"], self.optimisers["EBM_opt"], self.params["EBM_params"])
        self.params["GEN_params"], self.opt_states["GEN_opt_state"] = self.update_params(grad_gen, self.opt_states["GEN_opt_state"], self.optimisers["GEN_opt"], self.params["GEN_params"])
        
        return ebm_loss.mean() + gen_loss.mean()
    
    def validation_step(self, x):

        ebm_params = self.params["EBM_params"]
        gen_params = self.params["GEN_params"]

        return loss_fcn_EBM(ebm_params, x).mean() + loss_fcn_GEN(gen_params, x).mean()
    
    def update_params(self, grad, opt_state, optimiser, params):

        updates, new_state = optimiser.update(grad, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, new_state




        

