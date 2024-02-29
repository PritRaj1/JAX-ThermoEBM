import sys
import os

# os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="False"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]="0.7"
# os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
# os.environ["XLA_FLAGS"]="--xla_gpu_strict_conv_algorithm_picker=false"

sys.path.append("..")
import torch
import optax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tqdm
from torch.utils.data import DataLoader

from src.pipeline.LatentEBM_Trainer import Trainer
from src.utils.helper_functions import parse_input_file, get_data

# print(os.environ["XLA_FLAGS"], 
#       os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"], 
#       os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"], 
#       os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"])

print(f"Device: {jax.default_backend()}")

config = parse_input_file("hyperparams.input")

dataset, val_dataset, config["IMAGE_DIM"] = get_data(config["DATASET"])

# Convert the config values to the correct type
config.pop("DATASET")
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

for key, value in config.items():
    print(f"{key}: {value}")

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, range(config["NUM_TRAIN_DATA"]))
val_data = torch.utils.data.Subset(val_dataset, range(config["NUM_VAL_DATA"]))

# Split dataset
test_loader = DataLoader(train_data, batch_size=config["BATCH_SIZE"], shuffle=True)
val_loader = DataLoader(val_data, batch_size=config["BATCH_SIZE"], shuffle=False)

rng = jax.random.PRNGKey(0)

Trainer = Trainer(config, "logs/")
EBM_list, GEN_list = Trainer.setup()

# Train the model
tqdm_bar = tqdm.tqdm(range(config["NUM_EPOCHS"]))
for epoch in tqdm_bar:
    train_loss = 0

    for batch in test_loader:
        x = jnp.array(batch[0].numpy())
        x = jnp.transpose(x, (0, 3, 2, 1)) # NHWC -> NCHW
        loss, EBM_list, GEN_list = Trainer.train(x, epoch, EBM_list, GEN_list)
        train_loss += loss

    tqdm_bar.set_postfix({"train_loss": train_loss})

    if epoch % config["VAL_EVERY"] == 0:
        val_loss = 0
        for batch in val_loader:
            x = jnp.array(batch[0].numpy())
            val_loss += Trainer.validate(x, epoch, EBM_list, GEN_list)
        tqdm_bar.set_postfix({"train_loss": train_loss, "val_loss": val_loss})

    # Profile flops in first epoch
    if epoch == 0:
        Trainer.profile_flops(x)

# Generate an image
generated_image = Trainer.generate()

# Plot the generated image
plt.figure()
plt.imshow(generated_image)
plt.savefig("generated_image.png")
