import torch
import jax
import jax.numpy as jnp
import optax
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

# Metrics
from pypapi import events, papi_high as high
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.mifid import MemorizationInformedFrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.pipeline.pipeline_steps import get_losses, updara_params
from src.models.PriorModel import EBM
from src.models.GeneratorModel import GEN
from src.MCMC_Samplers.sample_distributions import sample_p0, sample_prior
from src.pipeline.loss_fcn import TI_GEN_loss_fcn

class Trainer():
    def __init__(self, config, log_path):
        self.key = jax.random.PRNGKey(0)

        # for key, value in config.items():
        #     try:
        #         config[key] = jnp.int32(value)
        #     except:
        #         try:
        #             config[key] = jnp.float32(value)
        #         except:
        #             try:
        #                 config[key] = eval(value)
        #             except:
        #                 pass

        self.config = config

        self.init_EBM()
        self.init_GEN()
        self.init_temps()

        self.tb_writer = SummaryWriter(log_path)
        self.csv_logger = None

        self.fid = FrechetInceptionDistance(feature=64, normalize=True) # FID metric
        self.mifid = MemorizationInformedFrechetInceptionDistance(feature=64, normalize=True) # MI-FID metric
        self.kid = KernelInceptionDistance(feature=64, subset_size=config['BATCH_SIZE'], normalize=True) # KID metric
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', normalize=True) # LPIPS metric

    def init_EBM(self):
        EBM_init_key = jax.random.PRNGKey(0)

        self.key, z_init = sample_p0(
                                    self.key, 
                                    self.config['p0_SIGMA'], 
                                    self.config['BATCH_SIZE'], 
                                    self.config['NUM_Z']
                                )
        
        # Create the EBM model
        EBM_model = EBM(
            hidden_units=self.config['EBM_FEATURE_DIM'], 
            output_dim=self.config['EMB_OUT_SIZE'],
            leak_coef=self.config['EBM_LEAK']
            )

        # Initialise the EBM model
        self.EBM_params = EBM_model.init(EBM_init_key, z_init)
        self.EBM_fwd = EBM_model.apply

        # Initialise the EBM optimiser
        E_schedule = optax.exponential_decay(
            init_value=self.config['E_LR'], 
            transition_steps=self.config['E_STEPS'], 
            decay_rate=self.config['E_GAMMA']
            )
        self.EBM_optimiser = optax.adam(learning_rate=E_schedule)

        del EBM_init_key

    def init_GEN(self):
        GEN_init_key = jax.random.PRNGKey(0)

        # Create the Generator model
        GEN_model = GEN(
            feature_dim=self.config['GEN_FEATURE_DIM'], 
            output_dim=self.config['CHANNELS'],
            image_dim=self.config['IMAGE_DIM'],
            leak_coef=self.config['GEN_LEAK']
            )
        
        self.key, z_init = sample_p0(
                            self.key, 
                            self.config['p0_SIGMA'], 
                            self.config['BATCH_SIZE'], 
                            self.config['NUM_Z']
                        )

        # Initialise the Generator model
        self.GEN_params = GEN_model.init(GEN_init_key, z_init)
        self.GEN_fwd = GEN_model.apply

        # Initialise the Generator optimiser
        G_schedule = optax.exponential_decay(
            init_value=self.config['G_LR'], 
            transition_steps=self.config['G_STEPS'], 
            decay_rate=self.config['G_GAMMA']
            )
        self.GEN_optimiser = optax.adam(learning_rate=G_schedule)

        del GEN_init_key

    def init_temps(self):
        
        if self.config['TEMP_POWER'] >= 1:
            print("Using temperature schedule with power: {}".format(self.config['TEMP_POWER']))
            self.temp = {
                "schedule": jnp.linspace(0, 1, self.config['NUM_TEMPS'])**self.config['TEMP_POWER'],
            }

        else:
            print("Using no thermodynamic integration, defaulting to Vanilla Model")
            self.temp = {
                "schedule": jnp.array([1]),
            }

    def train(self, x, epoch):

        # Get the losses
        self.key, loss_ebm, grad_ebm, loss_gen, grad_gen = get_losses(
            self.key,
            x,
            self.EBM_fwd,
            self.EBM_params,
            self.GEN_fwd,
            self.GEN_params,
            self.config['LKHOOD_SIGMA'],
            self.config['p0_SIGMA'],
            self.config['E_STEP_SIZE'],
            self.config['E_SAMPLE_STEPS'],
            self.config['G_STEP_SIZE'],
            self.config['G_SAMPLE_STEPS'],
            self.config['BATCH_SIZE'],
            self.config['NUM_Z'],
            self.temp['schedule']
        )

        # Update the parameters
        self.EBM_params, self.EBM_opt_state, self.GEN_params, self.GEN_opt_state = updara_params(
            self.EBM_params,
            self.EBM_opt_state,
            self.GEN_params,
            self.GEN_opt_state,
            grad_ebm,
            grad_gen,
            self.EBM_optimiser,
            self.GEN_optimiser
        )

        total_loss = loss_ebm.mean() + loss_gen.mean()

        self.tb_writer.add_scalar('train_Loss/EBM', loss_ebm.mean(), epoch)
        self.tb_writer.add_scalar('train_Loss/GEN', loss_gen.mean(), epoch)
        self.tb_writer.add_scalar('train_total_loss', total_loss, epoch)
        self.tb_writer.add_scalar('train_var_grad/EBM', jnp.var(grad_ebm), epoch)
        self.tb_writer.add_scalar('train_var_grad/GEN', jnp.var(grad_gen), epoch)

        return total_loss
    
    def validate(self, x, epoch):
            
            # Get the losses
            self.key, loss_ebm, grad_ebm, loss_gen, grad_gen = get_losses(
                self.key,
                x,
                self.EBM_fwd,
                self.EBM_params,
                self.GEN_fwd,
                self.GEN_params,
                self.config['LKHOOD_SIGMA'],
                self.config['p0_SIGMA'],
                self.config['E_STEP'],
                self.config['E_SAMPLE_STEPS'],
                self.config['G_STEP_SIZE'],
                self.config['G_SAMPLE_STEPS'],
                self.config['BATCH_SIZE'],
                self.config['NUM_Z'],
                self.temp['schedule']
            )
    
            total_loss = loss_ebm.mean() + loss_gen.mean()
    
            self.tb_writer.add_scalar('val_Loss/EBM', loss_ebm.mean(), epoch)
            self.tb_writer.add_scalar('val_Loss/GEN', loss_gen.mean(), epoch)
            self.tb_writer.add_scalar('val_total_loss', total_loss, epoch)
            self.tb_writer.add_scalar('val_var_grad/EBM', jnp.var(grad_ebm), epoch)
            self.tb_writer.add_scalar('val_var_grad/GEN', jnp.var(grad_gen), epoch)

            self.key, generated_data = self.generate()
            self.log_image_metrics(x, generated_data, epoch)
    
            return total_loss
    
    def generate(self):
        key, z_prior = sample_prior(
                                self.key, 
                                self.config['p0_SIGMA'], 
                                self.config['BATCH_SIZE'], 
                                self.config['NUM_Z']
                            )
        
        x_pred = self.GEN_fwd(self.GEN_params, jax.lax.stop_gradient(z_prior))

        return key, x_pred
    
    def log_image_metrics(self, x, x_pred, epoch):
        
        x = torch.from_numpy(x)
        x_pred = torch.from_numpy(x_pred)

        # Convert for [-1, 1] to [0, 1], image probailities
        x_metric = ((x + 1) / 2).reshape(-1, x.shape[1], x.shape[2], x.shape[3])
        gen_metric = (x_pred + 1) / 2

        # FID score
        self.fid.update(x_metric, real=True)
        self.fid.update(gen_metric, real=False)
        fid_score = self.fid.compute()

        # MI-FID score
        self.mifid.update(x_metric, real=True)
        self.mifid.update(gen_metric, real=False)
        mifid_score = self.mifid.compute()

        # KID score
        self.kid.update(x_metric, real=True)
        self.kid.update(gen_metric, real=False)
        kid_score = self.kid.compute()[0]

        # LPIPS score
        lpips_score = self.lpips(x_metric, gen_metric)

        self.tb_writer.add_scalar('val_FID', fid_score, epoch)
        self.tb_writer.add_scalar('val_MI-FID', mifid_score, epoch)
        self.tb_writer.add_scalar('val_KID', kid_score, epoch)
        self.tb_writer.add_scalar('val_LPIPS', lpips_score, epoch)

        # Log a grid of 4x4 images
        grid = make_grid(x_pred[:16], nrow=4)
        self.tb_writer.add_image('Generated Images', grid, epoch)

    def profile_flops(self, x):

        # Profile FLOPS for computing loss
        high.start_counters([events.PAPI_FP_OPS,])

        key, loss = TI_GEN_loss_fcn(
                                    self.key, 
                                    x, 
                                    self.EBM_fwd, 
                                    self.EBM_params, 
                                    self.GEN_fwd, 
                                    self.GEN_params, 
                                    self.config['LKHOOD_SIGMA'], 
                                    self.config['p0_SIGMA'], 
                                    self.config['G_STEP_SIZE'], 
                                    self.config['G_SAMPLE_STEPS'], 
                                    self.config['BATCH_SIZE'], 
                                    self.config['NUM_Z'], 
                                    self.temp['schedule']
                                )
        
        flops = high.stop_counters()

        self.tb_writer.add_scalar('FLOPS/GEN_loss', flops, 0)



        