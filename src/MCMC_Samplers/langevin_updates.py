import configparser

from src.MCMC_Samplers.sample_distributions import sample_prior
from src.MCMC_Samplers.grad_log_probs import prior_grad_log, posterior_grad_log

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

prior_steps = int(parser["MCMC"]["E_SAMPLE_STEPS"])
prior_s = float(parser["MCMC"]["E_STEP_SIZE"])
posterior_steps = int(parser["MCMC"]["G_SAMPLE_STEPS"])
posterior_s = float(parser["MCMC"]["G_STEP_SIZE"])

def langevin_prior(z, noise, EBM_params, EBM_fwd):

    # Compute the gradient of the log prior
    grad_f = prior_grad_log(z, EBM_params, EBM_fwd)
    
    # Update z_prior
    new_z = z + prior_s * prior_s * grad_f + noise

    return new_z, None

def langevin_posterior(z, noise, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd):

    # Compute the gradient of the log posterior
    grad_f = posterior_grad_log(z, x, t, EBM_params, GEN_params, EBM_fwd, GEN_fwd)

    # Update z_posterior
    new_z = z + posterior_s * posterior_s * grad_f + noise
    
    return new_z, None