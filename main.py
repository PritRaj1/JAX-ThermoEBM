import jax
from jax import config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
from functools import partial
import os
import torch
import configparser
import tqdm

from src.pipeline.initialise import *
from src.pipeline.batch_steps import train_step, val_step
from src.pipeline.generate import generate
from src.metrics.get_metrics import profile_image
from src.utils.helper_functions import get_data, make_grid, NumpyLoader

# Set plot styling
rc("font", **{"family": "serif", "serif": ["Computer Modern"]}, size=12)
rc("text", usetex=True)

os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.8"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"] = "platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"

print(f"Device: {jax.default_backend()}")
key = jax.random.PRNGKey(0)

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

data_set_name = parser["PIPELINE"]["DATASET"]
num_train_data = int(parser["PIPELINE"]["NUM_TRAIN_DATA"])
num_val_data = int(parser["PIPELINE"]["NUM_VAL_DATA"])
batch_size = int(parser["PIPELINE"]["BATCH_SIZE"])
num_epochs = int(parser["PIPELINE"]["NUM_EPOCHS"])

dataset, val_dataset = get_data(data_set_name)

# Take a subset of the dataset to ease computation
train_data = torch.utils.data.Subset(dataset, range(num_train_data))
val_data = torch.utils.data.Subset(val_dataset, range(num_val_data))

# Split dataset
train_loader = NumpyLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = NumpyLoader(val_data, batch_size=batch_size, shuffle=False)

# Stack into jnp array for scanning 
# train_x = np.stack([x for x, _ in train_loader])
val_x = np.stack([x for x, _ in val_loader])

# Extract all x for image comparisons
image_real = np.array([val_data[i][0] for i in range(100)])

key, EBM_params, EBM_fwd = init_EBM(key)
key, GEN_params, GEN_fwd = init_GEN(key)

EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

temp_schedule = init_temp_schedule()

params_tup = (EBM_params, GEN_params)
fwd_fcn_tup = (EBM_fwd, GEN_fwd)
optimiser_tup = (EBM_optimiser, GEN_optimiser)
opt_state_tup = (EBM_opt_state, GEN_opt_state)

log_path = f"logs/{data_set_name}/{temp_schedule[0]}"
os.makedirs(f"images/{data_set_name}", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Output number of parameters of generator
EBM_param_count = sum(x.size for x in jax.tree_util.tree_leaves(EBM_params))
GEN_param_count = sum(x.size for x in jax.tree_util.tree_leaves(GEN_params))
print(f"Number of parameters in generator: {GEN_param_count}")
print(f"Number of parameters in EBM: {EBM_param_count}")

# Preload the functions with immutable arguments
loaded_train_step = partial(train_step, optimiser_tup=optimiser_tup, fwd_fcn_tup=fwd_fcn_tup, temp_schedule=temp_schedule)
loaded_val_step = partial(val_step, fwd_fcn_tup=fwd_fcn_tup, temp_schedule=temp_schedule)
loaded_generate = partial(generate, num_images=len(image_real), fwd_fcn_tup=fwd_fcn_tup)

# Jit the functions
jit_train_step = jax.jit(loaded_train_step)
jit_val_step = jax.jit(loaded_val_step)

# def train_batches(carry, x):
#     key, params_tup, opt_state_tup = carry
    
#     key, params_tup, opt_state_tup, loss, var = jit_train_step(
#         key, x, params_tup, opt_state_tup
#     )
#     return (key, params_tup, opt_state_tup), (loss, var)

def val_batches(carry, x, params_tup):
    key = carry
    
    key, loss, var = jit_val_step(
        key, x, params_tup
    )
    return (key), (loss, var)

# Train the model
tqdm_bar = tqdm.tqdm(range(num_epochs))
for epoch in tqdm_bar:

    # Train - cannot scan due to large param count
    train_bar = tqdm.tqdm(train_loader, leave=False)
    train_loss = 0
    train_grad_var = 0
    for x, _ in train_bar:
        key, params_tup, opt_state_tup, train_loss, train_grad_var = jit_train_step(
            key, x, params_tup, opt_state_tup
        )
        train_bar.set_description(f"Train Loss: {train_loss}, Train Grad Var: {train_grad_var}")
        train_loss += train_loss
        train_grad_var += train_grad_var

    # train_loss = train_loss / len(train_loader)
    # train_grad_var = train_grad_var / len(train_loader)

    # Validate 
    key, (val_loss, val_grad_var) = jax.lax.scan(
        f = partial(val_batches, params_tup=params_tup),
        init = key,
        xs = val_x
    )

    val_loss = val_loss.sum()
    val_grad_var = val_grad_var.sum()

    # Profile generative capacity
    key, fake_images = loaded_generate(key, params_tup)
    fid, mifid, kid = profile_image(image_real, fake_images)

    random_indices = np.random.choice(len(fake_images), 4, replace=False)
    four_fake = np.array([fake_images[i] for i in random_indices])
    four_real = np.array([image_real[i] for i in random_indices])

    fake_grid = make_grid(four_fake, n_row=2)
    real_grid = make_grid(four_real, n_row=2)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(fake_grid)
    ax[0].set_title("Generated Images")
    ax[0].axis("off")
    ax[1].imshow(real_grid)
    ax[1].set_title("Real Images")
    ax[1].axis("off")
    plt.suptitle(
        "Epoch: {} \n\n FID: {:.2f}, MI-FID: {:.2f}, KID: {:.2f}".format(
            epoch, fid, mifid, kid
        )
    )
    plt.tight_layout()
    plt.savefig(f"images/{data_set_name}/{epoch}.png", dpi=500)
