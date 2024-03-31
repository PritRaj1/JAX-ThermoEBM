import jax
import configparser

from src.pipeline.loss_computation.loss_helper_fcns import mean_EBMloss, mean_GENloss

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])

def vanilla_loss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd):
    """Returns - log[ p(x | θ) ]"""
    ebm_loss, z_posterior = mean_EBMloss(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd)
    gen_loss = mean_GENloss(x, z_posterior, GEN_params, GEN_fwd)
    return - (ebm_loss + gen_loss)