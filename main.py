import sys; sys.path.append('..')
import torch
import optax
import jax
import matplotlib.pyplot as plt
import tqdm

from src.pipeline.train_state import Train_State
from src.models.PriorModel import EBM
from src.models.GeneratorModel import GEN
from src.MCMC_Samplers.latent_prior_MCMC import prior_sampler
from src.MCMC_Samplers.latent_posterior_sampler import posterior_sampler
from src.utils.helper_functions import parse_input_file, get_data
from src.pipeline.sample_generate_fcns import generate

config = parse_input_file('hyperparams.input')

dataset, val_dataset, config['IMAGE_DIM'] = get_data(config['DATASET'])

print(config)

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, config['NUM_TRAIN_DATA'])
val_data = torch.utils.data.Subset(val_dataset, config['NUM_VAL_DATA'])

rng = jax.random.PRNGKey(0)

# Create the EBM model
EBM_model = EBM(
    hidden_dim=config['EBM_FEATURE_DIM'], 
    output_dim=config['EMB_OUT_SIZE'],
    leak_coef=config['EBM_LEAK']
    )

# Create the Generator model
GEN_model = GEN(
    input_dim=config['EMB_OUT_SIZE'],
    feature_dim=config['GEN_FEATURE_DIM'], 
    output_dim=config['CHANNELS'],
    image_dim=config['IMAGE_DIM'],
    leak_coef=config['GEN_LEAK']
    )

# Create the optimisers
E_schedule = optax.exponential_decay(
    init_value=config['E_LR'], 
    decay_steps=config['E_STEPS'], 
    decay_rate=config['E_GAMMA']
    )
EBM_optimiser = optax.adam(learning_rate=E_schedule)

G_schedule = optax.exponential_decay(
    init_value=config['G_LR'], 
    decay_steps=config['G_STEPS'], 
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

    for i, batch in enumerate(train_data):
        train_loss += state.train_step(batch)

    print(f'Epoch: {epoch}, Loss: {train_loss / len(train_data)}')

    if epoch % config['VAL_EVERY'] == 0:
        val_loss = 0
        for i, batch in enumerate(val_data):
            val_loss += state.val_step(batch)

        print(f'Validation Loss: {val_loss / len(val_data)}')

generated_image = generate(state)[0]

# Plot the generated image
plt.figure()
plt.imshow(generated_image)
plt.savefig('generated_image.png')