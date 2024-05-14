import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

DATA_NAME = 'CIFAR10'

os.makedirs(f"results/{DATA_NAME}/evolutions", exist_ok=True)
os.makedirs(f"results/{DATA_NAME}/boxplots", exist_ok=True)
os.makedirs(f"results/{DATA_NAME}/relationships", exist_ok=True)

# Set plot styling
sns.set(font_scale=1.5)
sns.set_style("whitegrid", rc ={'text.usetex' : True, 'font.family' : 'serif', 'font.serif' : ['Computer Modern']})

NUM_EXPERIMENTS = 5
TEMPS = [0.1, 1, 3, 6, 10]
BATCH_SIZE = 75
OTHER_BATCHES = [25, 50, 75, 150]

# Colour maps for the different temperatures
temp_cmap = plt.get_cmap('coolwarm')
temp_colors = [temp_cmap(i) for i in np.linspace(0, 1, len(TEMPS))]
batch_cmap = plt.get_cmap('summer')
batch_colors = [batch_cmap(i) for i in np.linspace(0, 1, len(OTHER_BATCHES))]

#Dictionaries for train_loss, train_grad_var, val_loss, and val_FID at each temperature
dict_train_loss = {}
dict_train_grad = {}
dict_train_grad_var = {}
dict_val_loss = {}
dict_val_fid = {}
dict_val_kid = {}
dict_val_mifid = {}

for temp in TEMPS:

    # Load the data
    log_path = f"logs/{DATA_NAME}/p={temp}/batch={BATCH_SIZE}/"
    for i in range(NUM_EXPERIMENTS):
        df = pd.read_csv(f"{log_path}/experiment{i}.csv")

        # Create a dataframe with all the train loss
        if i == 0:
            all_train_loss = df["Train Loss"]
            all_train_grad = df["Train Grad"]
            all_train_grad_var = df["Train Grad Var"]
            all_val_loss = df["Val Loss"]
            all_fid = df["FID_inf"]
            all_mifid = df["MIFID_inf"]
            all_kid = df["KID_inf"]
            epochs = df["Epoch"]

        else:
            all_train_loss = pd.concat([all_train_loss, df["Train Loss"]], axis=1)
            all_train_grad = pd.concat([all_train_grad, df["Train Grad"]], axis=1)
            all_train_grad_var = pd.concat([all_train_grad_var, df["Train Grad Var"]], axis=1)
            all_val_loss = pd.concat([all_val_loss, df["Val Loss"]], axis=1)
            all_fid = pd.concat([all_fid, df['FID_inf']], axis=1)
            all_mifid = pd.concat([all_mifid, df['MIFID_inf']], axis=1)
            all_kid = pd.concat([all_kid, df['KID_inf']], axis=1)

    all_train_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_train_loss['epoch'] = epochs
    all_train_loss = all_train_loss.melt(id_vars='epoch', var_name='experiment', value_name='Train Loss')

    all_train_grad.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_train_grad['epoch'] = epochs
    all_train_grad = all_train_grad.melt(id_vars='epoch', var_name='experiment', value_name='Train Grad')

    all_train_grad_var.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_train_grad_var['epoch'] = epochs
    all_train_grad_var = all_train_grad_var.melt(id_vars='epoch', var_name='experiment', value_name='Train Grad Var')

    all_val_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_val_loss['epoch'] = epochs
    all_val_loss = all_val_loss.melt(id_vars='epoch', var_name='experiment', value_name='Val Loss')

    all_fid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_fid['epoch'] = epochs
    all_fid = all_fid.melt(id_vars='epoch', var_name='experiment', value_name='FID_inf')

    all_mifid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_mifid['epoch'] = epochs
    all_mifid = all_mifid.melt(id_vars='epoch', var_name='experiment', value_name='MIFID_inf')

    all_kid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_kid['epoch'] = epochs
    all_kid = all_kid.melt(id_vars='epoch', var_name='experiment', value_name='KID_inf')

    dict_train_loss[temp] = all_train_loss
    dict_train_grad[temp] = all_train_grad
    dict_train_grad_var[temp] = all_train_grad_var
    dict_val_loss[temp] = all_val_loss
    dict_val_fid[temp] = all_fid
    dict_val_kid[temp] = all_kid
    dict_val_mifid[temp] = all_mifid

# Only vanilla model, p=0, has batch size 25 and 150
dict_other_batch_train_loss = {}
dict_other_batch_train_grad = {}
dict_other_batch_train_grad_var = {}
dict_other_batch_val_loss = {}
dict_other_batch_fid = {}
dict_other_batch_mifid = {}
dict_other_batch_kid = {}

for batch_size in OTHER_BATCHES:
    # Load the data
    log_path = f"logs/{DATA_NAME}/p=0/batch={batch_size}/"
    for i in range(NUM_EXPERIMENTS):
        df = pd.read_csv(f"{log_path}/experiment{i}.csv")

        # Create a dataframe with all the train loss
        if i == 0:
            other_batch_train_loss = df["Train Loss"]
            other_batch_train_grad = df["Train Grad"]
            other_batch_train_grad_var = df["Train Grad Var"]
            other_batch_val_loss = df["Val Loss"]
            other_batch_fid = df["FID_inf"]
            other_batch_mifid = df["MIFID_inf"]
            other_batch_kid = df["KID_inf"]
            epochs = df["Epoch"]

        else:
            other_batch_train_loss = pd.concat([other_batch_train_loss, df["Train Loss"]], axis=1)
            other_batch_train_grad = pd.concat([other_batch_train_grad, df["Train Grad"]], axis=1)
            other_batch_train_grad_var = pd.concat([other_batch_train_grad_var, df["Train Grad Var"]], axis=1)
            other_batch_val_loss = pd.concat([other_batch_val_loss, df["Val Loss"]], axis=1)
            other_batch_fid = pd.concat([other_batch_fid, df['FID_inf']], axis=1)
            other_batch_mifid = pd.concat([other_batch_mifid, df['MIFID_inf']], axis=1)
            other_batch_kid = pd.concat([other_batch_kid, df['KID_inf']], axis=1)

    other_batch_train_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_train_loss['epoch'] = epochs
    other_batch_train_loss = other_batch_train_loss.melt(id_vars='epoch', var_name='experiment', value_name='Train Loss')

    other_batch_train_grad.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_train_grad['epoch'] = epochs
    other_batch_train_grad = other_batch_train_grad.melt(id_vars='epoch', var_name='experiment', value_name='Train Grad')

    other_batch_train_grad_var.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_train_grad_var['epoch'] = epochs
    other_batch_train_grad_var = other_batch_train_grad_var.melt(id_vars='epoch', var_name='experiment', value_name='Train Grad Var')

    other_batch_val_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_val_loss['epoch'] = epochs
    other_batch_val_loss = other_batch_val_loss.melt(id_vars='epoch', var_name='experiment', value_name='Val Loss')

    other_batch_fid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_fid['epoch'] = epochs
    other_batch_fid = other_batch_fid.melt(id_vars='epoch', var_name='experiment', value_name='FID_inf')

    other_batch_mifid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_mifid['epoch'] = epochs
    other_batch_mifid = other_batch_mifid.melt(id_vars='epoch', var_name='experiment', value_name='MIFID_inf')

    other_batch_kid.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_kid['epoch'] = epochs
    other_batch_kid = other_batch_kid.melt(id_vars='epoch', var_name='experiment', value_name='KID_inf')

    dict_other_batch_train_loss[batch_size] = other_batch_train_loss
    dict_other_batch_train_grad[batch_size] = other_batch_train_grad
    dict_other_batch_train_grad_var[batch_size] = other_batch_train_grad_var
    dict_other_batch_val_loss[batch_size] = other_batch_val_loss
    dict_other_batch_fid[batch_size] = other_batch_fid
    dict_other_batch_mifid[batch_size] = other_batch_mifid
    dict_other_batch_kid[batch_size] = other_batch_kid

### Evolutions ###

# 1. Train loss
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_train_loss[batch_size], x='epoch', y='Train Loss', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_train_loss[temp], x='epoch', y='Train Loss', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Train Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/train_loss.png")

# 2. Train gradient
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_train_grad[batch_size], x='epoch', y='Train Grad', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_train_grad[temp], x='epoch', y='Train Grad', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$\nabla_\theta \log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Train Gradient for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/train_grad.png")

# 3. Train gradient variance
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_train_grad_var[batch_size], x='epoch', y='Train Grad Var', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_train_grad_var[temp], x='epoch', y='Train Grad Var', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Average Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/train_grad_var.png")

# 4. Val loss
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_val_loss[batch_size], x='epoch', y='Val Loss', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_loss[temp], x='epoch', y='Val Loss', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Validation Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/val_loss.png")

# 5. FID
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_fid[batch_size], x='epoch', y='FID_inf', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_fid[temp], x='epoch', y='FID_inf', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Average ' + r"$\overline{FID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/fid.png")

# 6. KID
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_kid[batch_size], x='epoch', y='KID_inf', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_kid[temp], x='epoch', y='KID_inf', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{KID}_\infty$")
plt.title(f'Average ' + r"$\overline{KID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/kid.png")

# 7. MIFID
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_mifid[batch_size], x='epoch', y='MIFID_inf', label=label, color=batch_colors[OTHER_BATCHES.index(batch_size)])
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_mifid[temp], x='epoch', y='MIFID_inf', label=label, color=temp_colors[TEMPS.index(temp)])
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{MIFID}_\infty$")
plt.title(f'Average ' + r"$\overline{MIFID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/mifid.png")

# Only vanilla model, p=0, has batch size 25 and 150
for batch_size in OTHER_BATCHES:
    final_train_loss = dict_other_batch_train_loss[batch_size].groupby('experiment').last()
    final_train_grad = dict_other_batch_train_grad[batch_size].groupby('experiment').last()
    final_train_grad_var = dict_other_batch_train_grad_var[batch_size].groupby('experiment').last()
    final_val_loss = dict_other_batch_val_loss[batch_size].groupby('experiment').last()
    final_fid = dict_other_batch_fid[batch_size].groupby('experiment').last()
    final_kid = dict_other_batch_kid[batch_size].groupby('experiment').last()
    final_mifid = dict_other_batch_mifid[batch_size].groupby('experiment').last()

    final_train_loss['temp'] = r"$N_{batch}$ =" +f"{batch_size}"
    final_train_grad['temp'] = r"$N_{batch}$ =" +f"{batch_size}"
    final_train_grad_var['temp'] = r"$N_{batch}$ =" +f"{batch_size}"
    final_val_loss['temp'] = r"$N_{batch}$ =" +f"{batch_size}"
    final_fid['temp'] = r"$N_{batch}$ =" +f"{batch_size}"
    final_kid['temp'] = r"$N_{batch}$ =" +f"{batch_size}"
    final_mifid['temp'] = r"$N_{batch}$ =" +f"{batch_size}"

    if batch_size == OTHER_BATCHES[0]:
        all_final_train_loss = final_train_loss
        all_final_train_grad = final_train_grad
        all_final_train_grad_var = final_train_grad_var
        all_final_val_loss = final_val_loss
        all_final_fid = final_fid
        all_final_kid = final_kid
        all_final_mifid = final_mifid
    
    else:
        all_final_train_loss = pd.concat([all_final_train_loss, final_train_loss])
        all_final_train_grad = pd.concat([all_final_train_grad, final_train_grad])
        all_final_train_grad_var = pd.concat([all_final_train_grad_var, final_train_grad_var])
        all_final_val_loss = pd.concat([all_final_val_loss, final_val_loss])
        all_final_fid = pd.concat([all_final_fid, final_fid])
        all_final_kid = pd.concat([all_final_kid, final_kid])
        all_final_mifid = pd.concat([all_final_mifid, final_mifid])

### Boxplots ###

for temp in TEMPS:
    final_train_loss = dict_train_loss[temp].groupby('experiment').last()
    final_train_grad = dict_train_grad[temp].groupby('experiment').last()
    final_train_grad_var = dict_train_grad_var[temp].groupby('experiment').last()
    final_val_loss = dict_val_loss[temp].groupby('experiment').last()
    final_fid = dict_val_fid[temp].groupby('experiment').last()
    final_kid = dict_val_kid[temp].groupby('experiment').last()
    final_mifid = dict_val_mifid[temp].groupby('experiment').last()

    final_train_loss['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_train_grad['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_train_grad_var['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_val_loss['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_fid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_kid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_mifid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"

    all_final_train_loss = pd.concat([all_final_train_loss, final_train_loss])
    all_final_train_grad = pd.concat([all_final_train_grad, final_train_grad])
    all_final_train_grad_var = pd.concat([all_final_train_grad_var, final_train_grad_var])
    all_final_val_loss = pd.concat([all_final_val_loss, final_val_loss])
    all_final_fid = pd.concat([all_final_fid, final_fid])
    all_final_kid = pd.concat([all_final_kid, final_kid])
    all_final_mifid = pd.concat([all_final_mifid, final_mifid])

    
plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_train_loss, x='Train Loss', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Final Train Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_train_loss.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_train_grad, x='Train Grad', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\nabla_\theta \log(p(\mathbf{x}|\theta))$")
plt.title(f'Final Train Gradient for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_train_grad.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_train_grad_var, x='Train Grad Var', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Final Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_train_grad_var.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_val_loss, x='Val Loss', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Final Validation Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_val_loss.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_fid, x='FID_inf', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_fid.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_kid, x='KID_inf', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\overline{KID}_\infty$")
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_kid.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_mifid, x='MIFID_inf', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\overline{MIFID}_\infty$")
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/boxplots/final_mifid.png")

### Train gradient variance boxplots i.e. split up the data by temperature and batch size ### 

# Extract all 'Vanilla Model' data from all_final_train_grad_var
grad_var_bsize = all_final_train_grad_var[all_final_train_grad_var['temp'] == 'Vanilla Model']
for batch_size in OTHER_BATCHES:
    grad_var_bsize = pd.concat([grad_var_bsize, all_final_train_grad_var[all_final_train_grad_var['temp'] == r"$N_{batch}$ =" +f"{batch_size}"]])

# Extract all 'p' varying data from all_final_train_grad_var
grad_var_p = all_final_train_grad_var[all_final_train_grad_var['temp'] != 'Vanilla Model']
for batch_size in OTHER_BATCHES:
    grad_var_p = grad_var_p[grad_var_p['temp'] != r"$N_{batch}$ =" +f"{batch_size}"]
                            
# Variation with Batch Size using colour map for batch sizes
plt.figure(figsize=(13, 4))
sns.boxplot(data=grad_var_bsize, x='Train Grad Var', y='temp', fill=False, orient='h', hue='temp', palette=batch_colors)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p_\theta(\mathbf{x}))\right]$")
plt.ylabel(r"Batch Size")
plt.xlim(0.1, 1.2)
plt.title(f'Gradient Variance in the Vanilla Model for {NUM_EXPERIMENTS} Experiments')
plt.tight_layout()
plt.savefig(f"results/{DATA_NAME}/boxplots/grad_var_bsize.png")

# Variation with temperature
plt.figure(figsize=(13, 6))
sns.boxplot(data=grad_var_p, x='Train Grad Var', y='temp', fill=False, orient='h', hue='temp', palette=temp_colors)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p_\theta(\mathbf{x}))\right]$")
plt.ylabel(r"Temperature Power")
plt.xlim(0.1, 1.2)
plt.title(f'Gradient Variance in the Altered Model for {NUM_EXPERIMENTS} Experiments')
plt.tight_layout()
plt.savefig(f"results/{DATA_NAME}/boxplots/grad_var_p.png")


### Relationships ###

fid_var = pd.merge(all_final_train_grad_var, all_final_fid, on=['experiment', 'temp'])
mifid_var = pd.merge(all_final_train_grad_var, all_final_mifid, on=['experiment', 'temp'])
kid_var = pd.merge(all_final_train_grad_var, all_final_kid, on=['experiment', 'temp'])

# Plot final fid against variance, with error bars

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    plt.scatter(fid_var[fid_var['temp'] == label]['Train Grad Var'], fid_var[fid_var['temp'] == label]['FID_inf'], label=plot_label, marker='x', color=batch_colors[OTHER_BATCHES.index(batch_size)])
    reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    plt.scatter(fid_var[fid_var['temp'] == label]['Train Grad Var'], fid_var[fid_var['temp'] == label]['FID_inf'], label=plot_label, marker='x', color=temp_colors[TEMPS.index(temp)])

    if temp != 0.1 and temp != 0.3:
        reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])

# Remove outliers 
reg_points = pd.concat(reg_points)
mean = reg_points['FID_inf'].mean()
std = reg_points['FID_inf'].std()
reg_points = reg_points[~(reg_points['FID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='FID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{FID}_\infty$")
#plt.xscale('log')
# plt.ylim(240, 320)
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/fid_var.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    plt.scatter(mifid_var[mifid_var['temp'] == label]['Train Grad Var'], mifid_var[mifid_var['temp'] == label]['MIFID_inf'], label=plot_label, marker='x', color=batch_colors[OTHER_BATCHES.index(batch_size)])
    reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=$75' if temp != 0 else r'Vanilla Model, $N_{batch}=$75'
    plt.scatter(mifid_var[mifid_var['temp'] == label]['Train Grad Var'], mifid_var[mifid_var['temp'] == label]['MIFID_inf'], label=plot_label, marker='x', color=temp_colors[TEMPS.index(temp)])

    if temp != 0.1 and temp != 0.3:
        reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['MIFID_inf'].mean()
std = reg_points['MIFID_inf'].std()
reg_points = reg_points[~(reg_points['MIFID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='MIFID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{MIFID}_\infty$")
#plt.xscale('log')
# plt.ylim(500, 800)
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/mifid_var.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    plt.scatter(kid_var[kid_var['temp'] == label]['Train Grad Var'], kid_var[kid_var['temp'] == label]['KID_inf'], label=plot_label, marker='x', color=batch_colors[OTHER_BATCHES.index(batch_size)])
    reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    plt.scatter(kid_var[kid_var['temp'] == label]['Train Grad Var'], kid_var[kid_var['temp'] == label]['KID_inf'], label=plot_label, marker='x', color=temp_colors[TEMPS.index(temp)])

    if temp != 0.1 and temp != 0.3:
        reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['KID_inf'].mean()
std = reg_points['KID_inf'].std()
reg_points = reg_points[~(reg_points['KID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='KID_inf', data=reg_points, scatter=False, label='Regression', order=2)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{KID}_\infty$")
#plt.xscale('log')
# plt.ylim(0.02, 0.08)
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/kid_var.png")

fid_var = pd.merge(all_final_train_grad_var, all_final_fid, on=['experiment', 'temp'])
mifid_var = pd.merge(all_final_train_grad_var, all_final_mifid, on=['experiment', 'temp'])
kid_var = pd.merge(all_final_train_grad_var, all_final_kid, on=['experiment', 'temp'])

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=plot_label, color=batch_colors[OTHER_BATCHES.index(batch_size)])

    reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=plot_label, color=temp_colors[TEMPS.index(temp)])

    if temp != 0.1 and temp != 0.3:
        reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['FID_inf'].mean()
std = reg_points['FID_inf'].std()
reg_points = reg_points[~(reg_points['FID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='FID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{FID}_\infty$")
# plt.ylim(240, 320)
#plt.xscale('log')
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/fid_var_error.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=plot_label, capsize=2, color=batch_colors[OTHER_BATCHES.index(batch_size)])

    reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=plot_label, capsize=2, color=temp_colors[TEMPS.index(temp)])

    if temp != 0.1 and temp != 0.3:
        reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['MIFID_inf'].mean()
std = reg_points['MIFID_inf'].std()
reg_points = reg_points[~(reg_points['MIFID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='MIFID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{MIFID}_\infty$")
#plt.xscale('log')
# plt.ylim(500, 800)
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/mifid_var_error.png")

plt.figure(figsize=(16, 11))
reg_points = []
reg_points_two = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=plot_label, capsize=2, marker='x', color=batch_colors[OTHER_BATCHES.index(batch_size)])

    reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=plot_label, capsize=2, marker='x', color=temp_colors[TEMPS.index(temp)])

    if temp != 0.1 and temp != 0.3 and temp != 1:
        reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])
    else:
        reg_points_two.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['KID_inf'].mean()
std = reg_points['KID_inf'].std()
reg_points = reg_points[~(reg_points['KID_inf'] > mean + 1.5 * std)]

reg_points_two = pd.concat(reg_points_two)
mean = reg_points_two['KID_inf'].mean()
std = reg_points_two['KID_inf'].std()
reg_points_two = reg_points_two[~(reg_points_two['KID_inf'] > mean + 3 * std)]

sns.regplot(x='Train Grad Var', y='KID_inf', data=reg_points, scatter=False, label=r'Regression $p>1$', order=2)
sns.regplot(x='Train Grad Var', y='KID_inf', data=reg_points_two, scatter=False, label=r'Regression $0<p \leq 1 $', order=2)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p_\theta(\mathbf{x}))\right]$")
plt.ylabel(r"$\overline{KID}_\infty$")
#plt.xscale('log')
# plt.ylim(0.02, 0.08)
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/kid_var_error.png")

# Plot mean train grad var against p value 
mean_grad_var = []
std_grad_var = []
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    if temp != 0:
        grad_var = dict_train_grad_var[temp].groupby('experiment').last()['Train Grad Var'].values
        plt.errorbar(temp, np.mean(grad_var), yerr=np.std(grad_var), fmt='x', capsize=2, label=f'p={temp}', color=temp_colors[TEMPS.index(temp)])
plt.xlabel(r"Temperature Power, $p$")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/relationships/power_grad_var.png")


"""
### Extra plots ###

# Plot thirty temps and eighty steps on original kid vs grad var plot
thiry_temps_path = f"extra_logs/CelebA/temps=30/p=0.1/batch=75"
eighty_steps_path = f"extra_logs/CelebA/prior_mcmc=80/p=0.1/batch=75"

for i in range(3):
    thiry_temps_df = pd.read_csv(f"{thiry_temps_path}/experiment{i}.csv")
    eighty_steps_df = pd.read_csv(f"{eighty_steps_path}/experiment{i}.csv")

    thiry_temps_df['temp'] = 'p=0.1'
    eighty_steps_df['temp'] = 'p=0.1'

    if i == 0:
        all_thiry_temps_grad_var = thiry_temps_df['Train Grad Var']
        all_eighty_steps_grad_var = eighty_steps_df['Train Grad Var']
        all_thiry_temps_kid = thiry_temps_df['KID_inf']
        all_eighty_steps_kid = eighty_steps_df['KID_inf']
        epochs = df["Epoch"]

    else:
        all_thiry_temps_grad_var = pd.concat([all_thiry_temps_grad_var, thiry_temps_df['Train Grad Var']], axis=1)
        all_eighty_steps_grad_var = pd.concat([all_eighty_steps_grad_var, eighty_steps_df['Train Grad Var']], axis=1)
        all_thiry_temps_kid = pd.concat([all_thiry_temps_kid, thiry_temps_df['KID_inf']], axis=1)
        all_eighty_steps_kid = pd.concat([all_eighty_steps_kid, eighty_steps_df['KID_inf']], axis=1)


# Extract final values
final_thirty_temps_var = all_thiry_temps_grad_var.iloc[-1]
final_eighty_steps_var = all_eighty_steps_grad_var.iloc[-1]
final_thiry_temps_kid = all_thiry_temps_kid.iloc[-1]
final_eighty_steps_kid = all_eighty_steps_kid.iloc[-1]

plt.figure(figsize=(16, 11))
reg_points = []
reg_points_two = []
for batch_size in OTHER_BATCHES:
    label = r"$N_{batch}$ =" +f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, capsize=2, marker='x', color="black")

    reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'$p=0.1$' + r', $N_{batch}=75$ (previous)' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    
    if temp != 0.1:
        plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, capsize=2, marker='x', color="black")
    else:
        plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=plot_label, capsize=2, marker='x')
    if temp != 0.1 and temp != 0.3 and temp != 1:
        reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])
    else:
        reg_points_two.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['KID_inf'].mean()
std = reg_points['KID_inf'].std()
reg_points = reg_points[~(reg_points['KID_inf'] > mean + 1.5 * std)]

reg_points_two = pd.concat(reg_points_two)
mean = reg_points_two['KID_inf'].mean()
std = reg_points_two['KID_inf'].std()
reg_points_two = reg_points_two[~(reg_points_two['KID_inf'] > mean + 3 * std)]

# Plot error bars for new data
plt.errorbar(final_thirty_temps_var['Train Grad Var'].mean(), final_thiry_temps_kid['KID_inf'].mean(), xerr=final_thirty_temps_var['Train Grad Var'].std(), yerr=final_thiry_temps_kid['KID_inf'].std(), label=r"$p=0.1, N_t=30$", capsize=2, marker='x')
plt.errorbar(final_eighty_steps_var['Train Grad Var'].mean(), final_eighty_steps_kid['KID_inf'].mean(), xerr=final_eighty_steps_var['Train Grad Var'].std(), yerr=final_eighty_steps_kid['KID_inf'].std(), label=r"$p=0.1, K_\alpha=80$", capsize=2, marker='x')
             
sns.regplot(x='Train Grad Var', y='KID_inf', data=reg_points, scatter=False, order=2, color='gray')
sns.regplot(x='Train Grad Var', y='KID_inf', data=reg_points_two, scatter=False, order=2, color='gray')
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p_\theta(\mathbf{x}))\right]$")
plt.ylabel(r"$\overline{KID}_\infty$")
#plt.xscale('log')
# plt.ylim(0.02, 0.08)
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/kid_var_error_extra.png")

"""