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
from src.pipeline.loss_fcn import TI_EBM_loss_fcn, TI_GEN_loss_fcn
from src.utils.helper_functions import get_data, get_grad_var
from src.MCMC_Samplers.sample_distributions import sample_prior

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


EBM_loss_fcn = jax.vmap(TI_EBM_loss_fcn, in_axes=(None, 0, None, None, None, None, None))
GEN_loss_fcn = jax.vmap(TI_GEN_loss_fcn, in_axes=(None, 0, None, None, None, None, None))

EBM_params, EBM_fwd = init_EBM(key)
GEN_params, GEN_fwd = init_GEN(key, image_dim)

EBM_optimiser, EBM_opt_state = init_EBM_optimiser(EBM_params)
GEN_optimiser, GEN_opt_state = init_GEN_optimiser(GEN_params)

temp_schedule = init_temp_schedule()

@jax.jit
def generate(key, EBM_params, GEN_params):
    key, z = sample_prior(key, EBM_params, EBM_fwd)
    x_pred = GEN_fwd(GEN_params, jax.lax.stop_gradient(z))

    return x_pred

@jax.jit
def train_step(key, x, EBM_params, GEN_params, EBM_opt_state, GEN_opt_state):

    (loss_ebm, ebm_key), grad_ebm = value_and_grad(
        EBM_loss_fcn, argnums=2, has_aux=True
    )(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)
    (loss_gen, gen_key), grad_gen = value_and_grad(
        GEN_loss_fcn, argnums=3, has_aux=True
    )(ebm_key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)

    # Update the parameters
    EBM_updates, EBM_opt_state = EBM_optimiser.update(grad_ebm, EBM_opt_state)
    EBM_params = optax.apply_updates(EBM_params, EBM_updates)

    GEN_updates, GEN_opt_state = GEN_optimiser.update(grad_gen, GEN_opt_state)
    GEN_params = optax.apply_updates(GEN_params, GEN_updates)

    total_loss = loss_ebm + loss_gen
    grad_var = get_grad_var(grad_ebm, grad_gen)

    return (
        gen_key,
        EBM_params,
        GEN_params,
        EBM_opt_state,
        GEN_opt_state,
        total_loss,
        grad_var,
    )


@jax.jit
def validate(key, x, EBM_params, GEN_params):
    (loss_ebm, ebm_key), grad_ebm = value_and_grad(
        EBM_loss_fcn, argnums=3, has_aux=True
    )(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)
    (loss_gen, gen_key), grad_gen = value_and_grad(
        GEN_loss_fcn, argnums=3, has_aux=True
    )(ebm_key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)

    total_loss = loss_ebm + loss_gen
    grad_var = get_grad_var(grad_ebm, grad_gen)

    return gen_key, total_loss, grad_var


@jax.jit
def profile_flops(key, x, EBM_params, GEN_params):
    high.start_counters([events.PAPI_FP_OPS])

    (loss_ebm, ebm_key), grad_ebm = value_and_grad(
        EBM_loss_fcn, argnums=3, has_aux=True
    )(key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)
    (loss_gen, gen_key), grad_gen = value_and_grad(
        GEN_loss_fcn, argnums=3, has_aux=True
    )(ebm_key, x, EBM_params, GEN_params, EBM_fwd, GEN_fwd, temp_schedule)

    return high.stop_counters()[0]


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
        key, EBM_params, GEN_params, EBM_opt_state, GEN_opt_state, loss, grad_var = (
            train_step(key, x, EBM_params, GEN_params, EBM_opt_state, GEN_opt_state)
        )
        train_loss += loss
        train_grad_var += grad_var

    for batch in val_loader:
        x, _ = batch
        x = jnp.array(x.numpy())
        x = jnp.transpose(x, (0, 3, 2, 1))
        key, loss, grad_var = validate(key, x, EBM_params, GEN_params)
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

    # Profile flops in first epoch
    if epoch == 0:
        flops = profile_flops(key, x, EBM_params, GEN_params)
        print(f"FLOPS: {flops}")

# Generate an image
generated_image = generate(key, EBM_params, GEN_params)[0]

# Plot the generated image
plt.figure()
plt.imshow(generated_image)
plt.savefig("generated_image.png")
