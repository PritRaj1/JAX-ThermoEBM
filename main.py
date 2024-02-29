import jax
from jax import value_and_grad
import jax.numpy as jnp
import matplotlib.pyplot as plt
import optax
import torch
from torch.utils.data import DataLoader
import configparser
from pypapi import events, papi_high as high
import tqdm
from tensorboardX import SummaryWriter

from src.pipeline.initialise import *
from src.pipeline.train_val import train_step, validate, generate
from src.utils.helper_functions import get_data
from src.pipeline.metrics import profile_flops


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
test_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

EBM_params, EBM_fwd = init_EBM(key)
GEN_params, GEN_fwd = init_GEN(key, image_dim)

EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

temp_schedule = init_temp_schedule()

params_tup = (EBM_params, GEN_params)
fwd_fcn_tup = (EBM_fwd, GEN_fwd)
optimiser_tup = (EBM_optimiser, GEN_optimiser)
opt_state_tup = (EBM_opt_state, GEN_opt_state)

# Train the model
tqdm_bar = tqdm.tqdm(range(num_epochs))
for epoch in tqdm_bar:
    train_loss = 0
    train_grad_var = 0
    val_loss = 0
    val_grad_var = 0
    for batch in test_loader:
        x, _ = batch
        x = jnp.array(x.numpy())
        x = jnp.transpose(x, (0, 3, 2, 1))
        key, params_tup, opt_state_tup, loss, grad_var = train_step(
            key, x, params_tup, opt_state_tup, optimiser_tup, fwd_fcn_tup, temp_schedule
        )
        train_loss += loss
        train_grad_var += grad_var

    for batch in val_loader:
        x, _ = batch
        x = jnp.array(x.numpy())
        x = jnp.transpose(x, (0, 3, 2, 1))
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

    # Profile flops in final epoch
    if epoch == num_epochs - 1:
        flops = profile_flops(key, x, EBM_params, GEN_params)
        print(f"FLOPS: {flops}")

# Generate an image
generated_image = generate(key, params_tup, fwd_fcn_tup)

# Plot the generated image
plt.figure()
plt.imshow(generated_image)
plt.savefig("generated_image.png")
