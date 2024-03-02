import jax
from jax import value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import os
import torch
import configparser
import tqdm
from tensorboardX import SummaryWriter
import tensorflow as tf

from src.pipeline.initialise import *
from src.pipeline.train_val import train_step, validate, generate
from src.utils.helper_functions import get_data, NumpyLoader

# from src.pipeline.metrics import profile_flops

# tf.config.experimental.set_visible_devices([], "GPU")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"
# os.environ["JAX_TRACEBACK_FILTERING"]="off"

print(f"Device: {jax.default_backend()}")
key = jax.random.PRNGKey(0)

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

data_set_name = parser["PIPELINE"]["DATASET"]
num_train_data = int(parser["PIPELINE"]["NUM_TRAIN_DATA"])
num_val_data = int(parser["PIPELINE"]["NUM_VAL_DATA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
num_epochs = int(parser["PIPELINE"]["NUM_EPOCHS"])

dataset, val_dataset, image_dim = get_data(data_set_name)

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, range(num_train_data))
val_data = torch.utils.data.Subset(val_dataset, range(num_val_data))

# Split dataset
test_loader = NumpyLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = NumpyLoader(val_data, batch_size=batch_size, shuffle=False)

key, EBM_params, EBM_fwd = init_EBM(key)
key, GEN_params, GEN_fwd = init_GEN(key, image_dim)

EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

temp_schedule = init_temp_schedule()

params_tup = (EBM_params, GEN_params)
fwd_fcn_tup = (EBM_fwd, GEN_fwd)
optimiser_tup = (EBM_optimiser, GEN_optimiser)
opt_state_tup = (EBM_opt_state, GEN_opt_state)

log_path = f"logs/{data_set_name}/{temp_schedule[0]}"

# Output number of parameters of generator
EBM_param_count = sum(x.size for x in jax.tree_util.tree_leaves(EBM_params))
GEN_param_count = sum(x.size for x in jax.tree_util.tree_leaves(GEN_params))
print(f"Number of parameters in generator: {GEN_param_count}")
print(f"Number of parameters in EBM: {EBM_param_count}")

# Train the model
tqdm_bar = tqdm.tqdm(range(num_epochs))
for epoch in tqdm_bar:
    train_loss = 0
    train_grad_var = 0
    val_loss = 0
    val_grad_var = 0
    for batch in tqdm.tqdm(test_loader):
        x, _ = batch
        key, params_tup, opt_state_tup, loss, grad_var = train_step(
            key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule
        )
        train_loss += loss
        train_grad_var += grad_var

    for batch in val_loader:
        x, _ = batch
        key, loss, grad_var = validate(key, x, params_tup, fwd_fcn_tup, temp_schedule)
        val_loss += loss
        val_grad_var += grad_var

    tqdm_bar.set_postfix(
        {
            "Train Loss": train_loss / len(train_data),
            "Val Loss": val_loss / len(val_data),
            "Train Grad Var": train_grad_var / len(train_data),
            "Val Grad Var": val_grad_var / len(val_data),
        }
    )

    # # Profile flops in final epoch
    # if epoch == num_epochs - 1:
    #     profile_flops(key, x, params_tup, fwd_fcn_tup, temp_schedule, log_path)

# Generate an image
key, generated_image = generate(key, params_tup, fwd_fcn_tup)

# Plot the generated image
plt.figure()
plt.imshow(generated_image)
plt.axis("off")
plt.savefig("generated_image.png")
