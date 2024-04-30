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
sns.set(font_scale=1.1)
sns.set_style("whitegrid", rc ={'text.usetex' : True, 'font.family' : 'serif', 'font.serif' : ['Computer Modern']})

NUM_EXPERIMENTS = 5
TEMPS = [0, 1, 3, 10]
BATCH_SIZE = 75
OTHER_BATCHES = [25, 50]


dict_train_loss = {}
dict_train_grad_var = {}
dict_val_loss = {}
dict_fid = {}
dict_mifid = {}
dict_kid = {}

# Initialise empty dictionaries to store train_loss, train_grad_var, val_loss, and val_FID for each temperature
dict_train_loss = {}
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
            all_train_grad_var = df["Train Grad Var"]
            all_val_loss = df["Val Loss"]
            all_fid = df["FID_inf"]
            all_mifid = df["MIFID_inf"]
            all_kid = df["KID_inf"]
            epochs = df["Epoch"]

        else:
            all_train_loss = pd.concat([all_train_loss, df["Train Loss"]], axis=1)
            all_train_grad_var = pd.concat([all_train_grad_var, df["Train Grad Var"]], axis=1)
            all_val_loss = pd.concat([all_val_loss, df["Val Loss"]], axis=1)
            all_fid = pd.concat([all_fid, df['FID_inf']], axis=1)
            all_mifid = pd.concat([all_mifid, df['MIFID_inf']], axis=1)
            all_kid = pd.concat([all_kid, df['KID_inf']], axis=1)

    all_train_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    all_train_loss['epoch'] = epochs
    all_train_loss = all_train_loss.melt(id_vars='epoch', var_name='experiment', value_name='Train Loss')

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
    dict_train_grad_var[temp] = all_train_grad_var
    dict_val_loss[temp] = all_val_loss
    dict_val_fid[temp] = all_fid
    dict_val_kid[temp] = all_kid
    dict_val_mifid[temp] = all_mifid

# Only vanilla model, p=0, has batch size 25 and 150
dict_other_batch_train_loss = {}
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
            other_batch_train_grad_var = df["Train Grad Var"]
            other_batch_val_loss = df["Val Loss"]
            other_batch_fid = df["FID_inf"]
            other_batch_mifid = df["MIFID_inf"]
            other_batch_kid = df["KID_inf"]
            epochs = df["Epoch"]

        else:
            other_batch_train_loss = pd.concat([other_batch_train_loss, df["Train Loss"]], axis=1)
            other_batch_train_grad_var = pd.concat([other_batch_train_grad_var, df["Train Grad Var"]], axis=1)
            other_batch_val_loss = pd.concat([other_batch_val_loss, df["Val Loss"]], axis=1)
            other_batch_fid = pd.concat([other_batch_fid, df['FID_inf']], axis=1)
            other_batch_mifid = pd.concat([other_batch_mifid, df['MIFID_inf']], axis=1)
            other_batch_kid = pd.concat([other_batch_kid, df['KID_inf']], axis=1)

    other_batch_train_loss.columns = [f"experiment_{i}" for i in range(NUM_EXPERIMENTS)]
    other_batch_train_loss['epoch'] = epochs
    other_batch_train_loss = other_batch_train_loss.melt(id_vars='epoch', var_name='experiment', value_name='Train Loss')

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
    dict_other_batch_train_grad_var[batch_size] = other_batch_train_grad_var
    dict_other_batch_val_loss[batch_size] = other_batch_val_loss
    dict_other_batch_fid[batch_size] = other_batch_fid
    dict_other_batch_mifid[batch_size] = other_batch_mifid
    dict_other_batch_kid[batch_size] = other_batch_kid


# Plot the train loss
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_train_loss[batch_size], x='epoch', y='Train Loss', label=label)
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_train_loss[temp], x='epoch', y='Train Loss', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Train Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/train_loss.png")

# Plot the train grad var
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_train_grad_var[batch_size], x='epoch', y='Train Grad Var', label=label)
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_train_grad_var[temp], x='epoch', y='Train Grad Var', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Average Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/train_grad_var.png")

# Plot the val loss
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_val_loss[batch_size], x='epoch', y='Val Loss', label=label)
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_loss[temp], x='epoch', y='Val Loss', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Validation Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/val_loss.png")

# Plot the FID
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_fid[batch_size], x='epoch', y='FID_inf', label=label)
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_fid[temp], x='epoch', y='FID_inf', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Average ' + r"$\overline{FID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/fid.png")

# Plot the KID
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_kid[batch_size], x='epoch', y='KID_inf', label=label)
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_kid[temp], x='epoch', y='KID_inf', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{KID}_\infty$")
plt.title(f'Average ' + r"$\overline{KID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/kid.png")

# Plot the MIFID
plt.figure(figsize=(10, 6))
for batch_size in OTHER_BATCHES:
    label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    sns.lineplot(data=dict_other_batch_mifid[batch_size], x='epoch', y='MIFID_inf', label=label)
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' + r", $N_{batch}=75$" if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    sns.lineplot(data=dict_val_mifid[temp], x='epoch', y='MIFID_inf', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{MIFID}_\infty$")
plt.title(f'Average ' + r"$\overline{MIFID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/evolutions/mifid.png")

# Only vanilla model, p=0, has batch size 25 and 150
for batch_size in OTHER_BATCHES:
    final_train_loss = dict_other_batch_train_loss[batch_size].groupby('experiment').last()
    final_train_grad_var = dict_other_batch_train_grad_var[batch_size].groupby('experiment').last()
    final_val_loss = dict_other_batch_val_loss[batch_size].groupby('experiment').last()
    final_fid = dict_other_batch_fid[batch_size].groupby('experiment').last()
    final_kid = dict_other_batch_kid[batch_size].groupby('experiment').last()
    final_mifid = dict_other_batch_mifid[batch_size].groupby('experiment').last()

    final_train_loss['temp'] = r"Vanilla Model " + f"{batch_size}"
    final_train_grad_var['temp'] = r"Vanilla Model " + f"{batch_size}"
    final_val_loss['temp'] = r"Vanilla Model " + f"{batch_size}"
    final_fid['temp'] = r"Vanilla Model " + f"{batch_size}"
    final_kid['temp'] = r"Vanilla Model " + f"{batch_size}"
    final_mifid['temp'] = r"Vanilla Model " + f"{batch_size}"

    if batch_size == OTHER_BATCHES[0]:
        all_final_train_loss = final_train_loss
        all_final_train_grad_var = final_train_grad_var
        all_final_val_loss = final_val_loss
        all_final_fid = final_fid
        all_final_kid = final_kid
        all_final_mifid = final_mifid
    
    else:
        all_final_train_loss = pd.concat([all_final_train_loss, final_train_loss])
        all_final_train_grad_var = pd.concat([all_final_train_grad_var, final_train_grad_var])
        all_final_val_loss = pd.concat([all_final_val_loss, final_val_loss])
        all_final_fid = pd.concat([all_final_fid, final_fid])
        all_final_kid = pd.concat([all_final_kid, final_kid])
        all_final_mifid = pd.concat([all_final_mifid, final_mifid])

# Plot boxplot of the final loss and metrics for each temperature
for temp in TEMPS:
    final_train_loss = dict_train_loss[temp].groupby('experiment').last()
    final_train_grad_var = dict_train_grad_var[temp].groupby('experiment').last()
    final_val_loss = dict_val_loss[temp].groupby('experiment').last()
    final_fid = dict_val_fid[temp].groupby('experiment').last()
    final_kid = dict_val_kid[temp].groupby('experiment').last()
    final_mifid = dict_val_mifid[temp].groupby('experiment').last()

    final_train_loss['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_train_grad_var['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_val_loss['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_fid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_kid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_mifid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"

    all_final_train_loss = pd.concat([all_final_train_loss, final_train_loss])
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

fid_var = pd.merge(all_final_train_grad_var, all_final_fid, on=['experiment', 'temp'])
mifid_var = pd.merge(all_final_train_grad_var, all_final_mifid, on=['experiment', 'temp'])
kid_var = pd.merge(all_final_train_grad_var, all_final_kid, on=['experiment', 'temp'])


# Plot final fid against variance, with error bars

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"Vanilla Model " + f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    plt.scatter(fid_var[fid_var['temp'] == label]['Train Grad Var'], fid_var[fid_var['temp'] == label]['FID_inf'], label=plot_label, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    # std = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] > mean + 1.5 * std))]
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] < mean - 2 * std))]

    reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])
    # mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    # std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    # mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    # std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    # plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=label)
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    plt.scatter(fid_var[fid_var['temp'] == label]['Train Grad Var'], fid_var[fid_var['temp'] == label]['FID_inf'], label=plot_label, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    # std = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] > mean + 1.5 * std))]
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] < mean - 2 * std))]

    reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])
    # mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    # std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    # mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    # std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    # plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=label)

# Remove outliers 
reg_points = pd.concat(reg_points)
mean = reg_points['FID_inf'].mean()
std = reg_points['FID_inf'].std()
reg_points = reg_points[~(reg_points['FID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='FID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{FID}_\infty$")
#plt.xscale('log')
plt.ylim(240, 320)
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/fid_var.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"Vanilla Model " + f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    plt.scatter(mifid_var[mifid_var['temp'] == label]['Train Grad Var'], mifid_var[mifid_var['temp'] == label]['MIFID_inf'], label=plot_label, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    # std = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] > mean + 1.5 * std))]
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] < mean - 2 * std))]


    reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])
    # mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    # std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    # mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    # std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    # plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=label, capsize=2)
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=$75' if temp != 0 else r'Vanilla Model, $N_{batch}=$75'
    plt.scatter(mifid_var[mifid_var['temp'] == label]['Train Grad Var'], mifid_var[mifid_var['temp'] == label]['MIFID_inf'], label=plot_label, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    # std = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] > mean + 1.5 * std))]
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] < mean - 2 * std))]

    reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])
    # mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    # std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    # mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    # std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    # plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=label, capsize=2)

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['MIFID_inf'].mean()
std = reg_points['MIFID_inf'].std()
reg_points = reg_points[~(reg_points['MIFID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='MIFID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{MIFID}_\infty$")
#plt.xscale('log')
plt.ylim(500, 800)
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/mifid_var.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"Vanilla Model " + f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'
    plt.scatter(kid_var[kid_var['temp'] == label]['Train Grad Var'], kid_var[kid_var['temp'] == label]['KID_inf'], label=plot_label, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    # std = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] > mean + 1.5 * std))]
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] < mean - 2 * std))]


    reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])
    # mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    # std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    # mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    # std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    # plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=label, capsize=2, marker='x')
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'
    plt.scatter(kid_var[kid_var['temp'] == label]['Train Grad Var'], kid_var[kid_var['temp'] == label]['KID_inf'], label=plot_label, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    # std = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] > mean + 1.5 * std))]
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] < mean - 2 * std))]

    reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])
    # mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    # std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    # mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    # std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    # plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=label, capsize=2, marker='x')

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['KID_inf'].mean()
std = reg_points['KID_inf'].std()
reg_points = reg_points[~(reg_points['KID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='KID_inf', data=reg_points, scatter=False, label='Regression', order=2)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{KID}_\infty$")
#plt.xscale('log')
plt.ylim(0.02, 0.08)
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/kid_var.png")

fid_var = pd.merge(all_final_train_grad_var, all_final_fid, on=['experiment', 'temp'])
mifid_var = pd.merge(all_final_train_grad_var, all_final_mifid, on=['experiment', 'temp'])
kid_var = pd.merge(all_final_train_grad_var, all_final_kid, on=['experiment', 'temp'])

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"Vanilla Model " + f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=plot_label)

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    # std = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] > mean + 1.5 * std))]
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] < mean - 2 * std))]

    reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=plot_label)

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    # std = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] > mean + 1.5 * std))]
    # fid_var = fid_var[~((fid_var['temp'] == label) & (fid_var['FID_inf'] < mean - 2 * std))]

    reg_points.append(fid_var[fid_var['temp'] == label][['Train Grad Var', 'FID_inf']])

# Remove outliers
reg_points = pd.concat(reg_points)
mean = reg_points['FID_inf'].mean()
std = reg_points['FID_inf'].std()
reg_points = reg_points[~(reg_points['FID_inf'] > mean + 1.5 * std)]

sns.regplot(x='Train Grad Var', y='FID_inf', data=reg_points, scatter=False, label='Regression', order=3)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{FID}_\infty$")
plt.ylim(240, 320)
#plt.xscale('log')
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/fid_var_error.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"Vanilla Model " + f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=plot_label, capsize=2)

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    # std = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] > mean + 1.5 * std))]
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] < mean - 2 * std))]

    reg_points.append(mifid_var[mifid_var['temp'] == label][['Train Grad Var', 'MIFID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=plot_label, capsize=2)

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    # std = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] > mean + 1.5 * std))]
    # mifid_var = mifid_var[~((mifid_var['temp'] == label) & (mifid_var['MIFID_inf'] < mean - 2 * std))]

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
plt.ylim(500, 800)
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + ' against Gradient Variance')
plt.legend(loc='upper right')
plt.savefig(f"results/{DATA_NAME}/relationships/mifid_var_error.png")

plt.figure(figsize=(14, 6))
reg_points = []
for batch_size in OTHER_BATCHES:
    label = r"Vanilla Model " + f"{batch_size}"
    plot_label = r'Vanilla Model, $N_{batch}=$' + f'{batch_size}'

    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=plot_label, capsize=2, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    # std = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] > mean + 1.5 * std))]
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] < mean - 2 * std))]

    reg_points.append(kid_var[kid_var['temp'] == label][['Train Grad Var', 'KID_inf']])

for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    plot_label = r'p=' + f'{temp}' + r', $N_{batch}=75$' if temp != 0 else r'Vanilla Model, $N_{batch}=75$'

    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=plot_label, capsize=2, marker='x')

    # # Remove outliers for regression # (2 stds from the mean)
    # mean = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    # std = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] > mean + 1.5 * std))]
    # kid_var = kid_var[~((kid_var['temp'] == label) & (kid_var['KID_inf'] < mean - 2 * std))]

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
plt.ylim(0.02, 0.08)
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
        plt.errorbar(temp, np.mean(grad_var), yerr=np.std(grad_var), fmt='x', capsize=2)
plt.xlabel(r"Temperature Power, $p$")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/{DATA_NAME}/relationships/power_grad_var.png")


