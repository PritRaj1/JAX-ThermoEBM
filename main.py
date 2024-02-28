import sys; sys.path.append('..')
import torch
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader

from pipeline.train_state_broken import Train_State
from src.models.PriorModel import EBM
from src.models.GeneratorModel import GEN
from src.MCMC_Samplers.latent_prior_MCMC import prior_sampler
from src.MCMC_Samplers.latent_posterior_sampler import posterior_sampler
from src.utils.helper_functions import parse_input_file, get_data
from src.pipeline.sample_generate_fcns import generate

config = parse_input_file('hyperparams.input')

dataset, val_dataset, config['IMAGE_DIM'] = get_data(config['DATASET'])

# Convert the config values to the correct type
config.pop('DATASET')
for key, value in config.items():
    try:
        config[key] = int(value)
    except:
        try:
            config[key] = float(value)
        except:
            try:
                config[key] = eval(value)
            except:
                pass

print(config)

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, range(config['NUM_TRAIN_DATA']))
val_data = torch.utils.data.Subset(val_dataset, range(config['NUM_VAL_DATA']))

# Split dataset
test_loader = DataLoader(train_data, batch_size=config['BATCH_SIZE'], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config['BATCH_SIZE'], shuffle=False)

rng = jax.random.PRNGKey(0)

# Create the EBM model
EBM_model = EBM(
    hidden_units=config['EBM_FEATURE_DIM'], 
    output_dim=config['EMB_OUT_SIZE'],
    leak_coef=config['EBM_LEAK']
    )

# Create the Generator model
GEN_model = GEN(
    feature_dim=config['GEN_FEATURE_DIM'], 
    output_dim=config['CHANNELS'],
    image_dim=config['IMAGE_DIM'],
    leak_coef=config['GEN_LEAK']
    )

# Create the optimisers
E_schedule = optax.exponential_decay(
    init_value=config['E_LR'], 
    transition_steps=config['E_STEPS'], 
    decay_rate=config['E_GAMMA']
    )
EBM_optimiser = optax.adam(learning_rate=E_schedule)

G_schedule = optax.exponential_decay(
    init_value=config['G_LR'], 
    transition_steps=config['G_STEPS'], 
    decay_rate=config['G_GAMMA']
    )
GEN_optimiser = optax.adam(learning_rate=G_schedule)

# Create the prior and posterior samplers
z_prior_sampler = prior_sampler(
    step_size=config['E_STEP'], 
    num_steps=config['E_SAMPLE_STEPS'], 
    p0_sigma=config['p0_SIGMA'], 
    num_z=config['Z_SAMPLES'], 
    batch_size=config['BATCH_SIZE']
    )

z_posterior_sampler = posterior_sampler(
    step_size=config['G_STEP'], 
    num_steps=config['G_SAMPLE_STEPS'], 
    lkhood_sigma=config['LKHOOD_SIGMA']
    )

# Create the training state
state = Train_State(
    key = rng,
    EBM_model = EBM_model,
    GEN_model = GEN_model,
    EBM_optimiser = EBM_optimiser,
    GEN_optimiser = GEN_optimiser,
    prior_sampler = z_prior_sampler,
    posterior_sampler = z_posterior_sampler,
    temperature_power = config['TEMP_POWER'],
    num_temps = config['NUM_TEMPS']
    )

# Train the model
tqdm_bar = tqdm.tqdm(range(config['NUM_EPOCHS']))
for epoch in tqdm_bar:
    train_loss = 0

    for batch in test_loader:
        x = jnp.array(batch[0].numpy())
        train_loss += state.training_step(x)

    tqdm_bar.set_postfix({'train_loss': train_loss})

    if epoch % config['VAL_EVERY'] == 0:
        val_loss = 0
        for batch in val_loader:
            x = jnp.array(batch[0].numpy())
            val_loss += state.validation_step(x)
        tqdm_bar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})

# Generate an image
generated_image = generate(state, rng)

# Plot the generated image
plt.figure()
plt.imshow(generated_image)
plt.savefig('generated_image.png')