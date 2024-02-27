import jax
import jnp

from src.pipeline.sample_generate_fcns import sample_z

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
            "EBM_opt": EBM_optimiser.init(self.params["EBM_params"]),
            "GEN_opt": GEN_optimiser.init(self.params["GEN_params"])
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
        

