import jax
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
import configparser
import tqdm

from src.pipeline.initialise import *
from src.pipeline.scan_batch import train_epoch, val_epoch
from src.pipeline.update_steps import generate
from src.utils.helper_functions import get_data, NumpyLoader

# from src.pipeline.metrics import profile_flops

# tf.config.experimental.set_visible_devices([], "GPU")
# os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.4"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["XLA_FLAGS"] = "--xla_gpu_strict_conv_algorithm_picker=false --xla_gpu_force_compilation_parallelism=1"
# os.environ["JAX_TRACEBACK_FILTERING"]="off"
os.environ["JAX_DEBUG_NANS"]="True"

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
os.makedirs('images', exist_ok=True)

# Output number of parameters of generator
EBM_param_count = sum(x.size for x in jax.tree_util.tree_leaves(EBM_params))
GEN_param_count = sum(x.size for x in jax.tree_util.tree_leaves(GEN_params))
print(f"Number of parameters in generator: {GEN_param_count}")
print(f"Number of parameters in EBM: {EBM_param_count}")

# Train the model
tqdm_bar = tqdm.tqdm(range(num_epochs))
for epoch in tqdm_bar:
    key, params_tup, opt_state_tup, train_loss, train_grad_var = train_epoch(
        key,
        test_loader,
        params_tup,
        opt_state_tup,
        optimiser_tup,
        fwd_fcn_tup,
        temp_schedule,
    )

    key, val_loss, val_grad_var = val_epoch(
        key, val_loader, params_tup, fwd_fcn_tup, temp_schedule
    )

    key, image = generate(key, params_tup, fwd_fcn_tup)
    image = np.array(image, dtype=np.float32) * 0.5 + 0.5

    plt.figure()
    plt.imshow(image, interpolation="nearest")
    plt.axis("off")
    plt.title(f"Epoch: {epoch}")
    plt.savefig(f"images/epoch_{epoch}.png", dpi=750)

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

# Cast from [-1, 1] to [0, 255], uint8
generated_image = (np.array(generated_image, dtype=np.float32) + 1) * 127.5

# Plot the generated image
plt.figure()
plt.imshow(generated_image, interpolation="nearest")
plt.axis("off")
plt.savefig("generated_image.png", dpi=1000)
