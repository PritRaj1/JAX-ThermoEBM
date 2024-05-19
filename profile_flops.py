import jax
import matplotlib.pyplot as plt
from matplotlib import rc
from functools import partial
import pandas as pd
import numpy as np
import configparser

from src.pipeline.initialise import *
from src.pipeline.pipeline_steps import train_step

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

num_temps = int(parser["TEMP"]["NUM_TEMPS"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
posterior_mcmc = int(parser["MCMC"]["G_SAMPLE_STEPS"])

"""Get number of flops to do validation"""

 # Initialise the pipeline
key = jax.random.PRNGKey(0)
key, EBM_params, EBM_fwd = init_EBM(key)
key, GEN_params, GEN_fwd = init_GEN(key)
EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

# Tuple up for cleanliness
params_tup = (EBM_params, GEN_params)
fwd_fcn_tup = (EBM_fwd, GEN_fwd)
optimiser_tup = (EBM_optimiser, GEN_optimiser)
opt_state_tup = (EBM_opt_state, GEN_opt_state)
del EBM_params, GEN_params, EBM_fwd, GEN_fwd, EBM_optimiser, GEN_optimiser, EBM_opt_state, GEN_opt_state

jit_train = jax.jit(partial(train_step, optimiser_tup=optimiser_tup, fwd_fcn_tup=fwd_fcn_tup))

test_x = np.random.randn(batch_size, 64, 64, 1)

# Compile the function
compiled_val = jit_train.lower(key, test_x, params_tup, opt_state_tup).compile()

# Get the FLOPs estimate from the cost analysis
flops = compiled_val.cost_analysis()[0]['flops']

print(f"Estimated FLOPs for jit_val: {flops}")

# Append to CSV
df = pd.DataFrame({"num_temps": [num_temps],"posterior_mcmc": [posterior_mcmc], "flops": [flops]})

df.to_csv("results/flops.csv", mode="a", header=False, index=False)
