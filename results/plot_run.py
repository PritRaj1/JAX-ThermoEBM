import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Set plot styling
sns.set(font_scale=1.1)
sns.set_style("whitegrid", rc ={'text.usetex' : True, 'font.family' : 'serif', 'font.serif' : ['Computer Modern']})

NUM_EXPERIMENTS = 3
TEMP = 1
BATCH_SIZE = 75

dict_train_loss = {}
dict_train_grad_var = {}
dict_val_loss = {}
dict_train_val_grad_var = {}
dict_fid = {}
dict_mifid = {}
dict_kid = {}

# Load the data
log_path = f"logs/CelebA/p={TEMP}/batch={BATCH_SIZE}/"
for i in range(NUM_EXPERIMENTS):
    df = pd.read_csv(f"{log_path}/experiment{i}.csv")

     # Create a dataframe with all the train loss
    if i == 0:
        all_train_loss = df[["Train Loss"]]
        all_train_grad_var = df[["Train Grad Var"]]
        all_val_loss = df[["Val Loss"]]
        all_train_val_grad_var = df[["Val Grad Var"]]
        all_fid = df[["FID_inf"]]
        all_mifid = df[["MIFID_inf"]]
        all_kid = df[["KID_inf"]]
        epochs = df["Epoch"]

    else:
        all_train_loss = pd.concat([all_train_loss, df["Train Loss"]], axis=1)
        all_train_grad_var = pd.concat([all_train_grad_var, df["Train Grad Var"]], axis=1)
        all_val_loss = pd.concat([all_val_loss, df["Val Loss"]], axis=1)
        all_train_val_grad_var = pd.concat([all_train_val_grad_var, df['Val Grad Var']], axis=1)
        all_fid = pd.concat([all_fid, df['FID_inf']], axis=1)
        all_mifid = pd.concat([all_mifid, df['MIFID_inf']], axis=1)
        all_kid = pd.concat([all_kid, df['KID_inf']], axis=1)

all_train_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_train_loss['epoch'] = epochs
all_train_loss = all_train_loss.melt(id_vars='epoch', var_name='experiment', value_name="Train Loss")

all_train_grad_var.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_train_grad_var['epoch'] = epochs
all_train_grad_var = all_train_grad_var.melt(id_vars='epoch', var_name='experiment', value_name="Train Grad Var")

all_val_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_val_loss['epoch'] = epochs
all_val_loss = all_val_loss.melt(id_vars='epoch', var_name='experiment', value_name="Val Loss")

all_train_val_grad_var.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_train_val_grad_var['epoch'] = epochs
all_train_val_grad_var = all_train_val_grad_var.melt(id_vars='epoch', var_name='experiment', value_name='Val Grad Var')

all_fid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_fid['epoch'] = epochs
all_fid = all_fid.melt(id_vars='epoch', var_name='experiment', value_name='FID_inf')

all_mifid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_mifid['epoch'] = epochs
all_mifid = all_mifid.melt(id_vars='epoch', var_name='experiment', value_name='MIFID_inf')

all_kid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
all_kid['epoch'] = epochs
all_kid = all_kid.melt(id_vars='epoch', var_name='experiment', value_name='KID_inf')

# Plot the train loss
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_train_loss, x='epoch', y="Train Loss")
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Train Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/train_loss.png")

# Plot the train grad var
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_train_grad_var, x='epoch', y='Train Grad Var')
plt.xlabel("Epoch")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Average Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/train_grad_var.png")

# Plot the val loss
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_val_loss, x='epoch', y='Val Loss')
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Validation Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/val_loss.png")

# Plot the val grad var
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_train_val_grad_var, x='epoch', y='Val Grad Var')
plt.xlabel("Epoch")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Average Validation Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/val_grad_var.png")

# Plot the FID
plt.figure(figsize=(10, 6))
sns.lineplot(data=all_fid, x='epoch', y='FID_inf')
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Average ' + r"$\overline{FID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/fid.png")
