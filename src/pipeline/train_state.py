import jax
import jax.numpy as jnp
import optax
# from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger

from src.pipeline.sample_generate_fcns import sample_z
from src.pipeline.loss_fcn import discretised_TI_loss_fcn, ebm_loss, gen_loss

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

        self_loss_fcns = {
            "EBM_loss_fcn": discretised_TI_loss_fcn(EBM_loss, gen=False),
            "GEN_loss_fcm": discretised_TI_loss_fcn(GEN_loss, gen=True)
        }

        self.optimisers = {
            "EBM_opt": EBM_optimiser.init(self.params["EBM_params"]),
            "GEN_opt": GEN_optimiser.init(self.params["GEN_params"])
        }

        self.opt_states = {
            "EBM_opt_state": self.optimisers["EBM_opt"].init(self.params["EBM_params"]),
            "GEN_opt_state": self.optimisers["GEN_opt"].init(self.params["GEN_params"])
        }

        self.samplers = {
            "prior": prior_sampler,
            "posterior": posterior_sampler
        }

        if temperature_power >= 1:
            print("Using temperature schedule with power: {}".format(temperature_power))
            self.temp = {
                "schedule": jnp.linspace(0, 1, num_temps)**temperature_power,
                "current": 0
            }

        else:
            print("Using no thermodynamic integration, defaulting to Vanilla Model")
            self.temp = {
                "schedule": jnp.array([1]),
                "current": 1
            }

        self.loggers = {
            "tb_writer": None,
            "csv_logger": None
        }

    @jax.jit
    def training_step(self, batch):

        x, _ = batch

        ebm_params = self.params["EBM_params"]
        gen_params = self.params["GEN_params"]

        loss_ebm = self.loss_fcns["EBM_loss_fcn"]
        loss_gen = self.loss_fcns["GEN_loss_fcm"]

        grad_ebm = jax.grad(loss_ebm)(ebm_params, x)
        grad_gen = jax.grad(loss_gen)(gen_params, x)

        self.params["EBM_params"], self.opt_states["EBM_opt_state"] = self.update_params(ebm_params, 
                                                                          grad_ebm, 
                                                                          self.optimisers["EBM_opt"], 
                                                                          self.opt_states["EBM_opt_state"])
        
        self.params["GEN_params"], self.opt_states["GEN_opt_state"] = self.update_params(gen_params,
                                                                            grad_gen,
                                                                            self.optimisers["GEN_opt"],
                                                                            self.opt_states["GEN_opt_state"])
        
        return loss_ebm.mean() + loss_gen.mean()
    
    @jax.jit
    def validation_step(self, batch):
        x, _ = batch

        ebm_params = self.params["EBM_params"]
        gen_params = self.params["GEN_params"]

        loss_ebm = self.loss_fcns["EBM_loss_fcn"]
        loss_gen = self.loss_fcns["GEN_loss_fcm"]

        return loss_ebm(ebm_params, x) + loss_gen(gen_params, x)


    def update_params(self, params, grads, optimiser, opt_state):

        updates, optstate = optimiser.update(grads, opt_state)
        new_params = optax.apply_updates(params, updates)

        return new_params, optstate




        

