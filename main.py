import jax
from jax import config
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
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
rc("font", **{"family": "serif", "serif": ["Computer Modern"]}, size=14)
rc("text", usetex=True)

# from src.pipeline.metrics import profile_flops

# tf.config.experimental.set_visible_devices([], "GPU")
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["XLA_FLAGS"] = ("--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1")
# os.environ['XLA_FLAGS'] = (
#     '--xla_gpu_enable_triton_softmax_fusion=true '
#     '--xla_gpu_triton_gemm_any=True '
#     '--xla_gpu_enable_async_collectives=true '
#     '--xla_gpu_enable_latency_hiding_scheduler=true '
#     '--xla_gpu_enable_highest_priority_async_stream=true '
# )
# os.environ["JAX_TRACEBACK_FILTERING"]="off"
# os.environ["JAX_DEBUG_NANS"]="True"
# config.update("jax_debug_nans", True)
# config.update('jax_disable_jit', True)
# config.update("jax_enable_x64", True)
# os.environ["JAX_CHECK_TRACER_LEAKS"] = "True"
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

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, range(num_train_data))
val_data = torch.utils.data.Subset(val_dataset, range(num_val_data))

# Split dataset
test_loader = NumpyLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = NumpyLoader(val_data, batch_size=batch_size, shuffle=False)

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
os.makedirs("images", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# Output number of parameters of generator
EBM_param_count = sum(x.size for x in jax.tree_util.tree_leaves(EBM_params))
GEN_param_count = sum(x.size for x in jax.tree_util.tree_leaves(GEN_params))
print(f"Number of parameters in generator: {GEN_param_count}")
print(f"Number of parameters in EBM: {EBM_param_count}")

jit_train_step = jax.jit(train_step, static_argnums=(4, 5))
jit_val_step = jax.jit(val_step, static_argnums=3)

# Train the model
tqdm_bar = tqdm.tqdm(range(num_epochs))
for epoch in tqdm_bar:

    epoch_loss = 0
    epoch_grad_var = 0

    batch_bar = tqdm.tqdm(test_loader, leave=False)
    for x, _ in batch_bar:  # tqdm.tqdm(test_loader):

        key, params_tup, opt_state_tup, batch_loss, batch_var = jit_train_step(
            key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule
        )

        epoch_loss += batch_loss
        epoch_grad_var += batch_var

        batch_bar.set_postfix(
            {
                "Train Loss": batch_loss,
                "Train Grad Var": batch_var,
            }
        )

    val_loss = 0
    val_grad_var = 0

    batch_bar = tqdm.tqdm(val_loader, leave=False)
    for x, _ in batch_bar:
        key, batch_loss, batch_var = jit_val_step(
            key, x, params_tup, fwd_fcn_tup, temp_schedule
        )

        val_loss += batch_loss
        val_grad_var += batch_var

        # Profile generative capacity
        key, fake_images = generate(key, params_tup, batch_size, fwd_fcn_tup)
        fid, mifid, kid = profile_image(x, fake_images)

        batch_bar.set_postfix(
            {
                "Val Loss": batch_loss,
                "Val Grad Var": batch_var,
                "FID": fid,
                "MI-FID": mifid,
                "KID": kid,
            }
        )

    fake_grid = make_grid(fake_images[:4], n_row=2)
    real_grid = make_grid(x[:4], n_row=2)

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(fake_grid)
    ax[0].set_title("Generated Image")
    ax[0].axis("off")
    ax[1].imshow(real_grid)
    ax[1].set_title("Real Image")
    ax[1].axis("off")
    plt.suptitle(f"Epoch: {epoch}\n FID: {fid}, MI-FID: {mifid}, KID: {kid}")
    plt.tight_layout()
    plt.savefig(f"images/{epoch}.png", dpi=500)
