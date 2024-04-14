import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

os.makedirs("results/evolutions", exist_ok=True)
os.makedirs("results/boxplots", exist_ok=True)
os.makedirs("results/relationships", exist_ok=True)

# Set plot styling
sns.set(font_scale=1.1)
sns.set_style("whitegrid", rc ={'text.usetex' : True, 'font.family' : 'serif', 'font.serif' : ['Computer Modern']})

NUM_EXPERIMENTS = 5
TEMPS = [0, 1, 3, 10]
BATCH_SIZE = 75

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
    log_path = f"logs/CelebA/p={temp}/batch={BATCH_SIZE}/"
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

# Plot the train loss
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' if temp != 0 else r'Vanilla Model'
    sns.lineplot(data=dict_train_loss[temp], x='epoch', y='Train Loss', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Train Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/evolutions/train_loss.png")

# Plot the train grad var
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' if temp != 0 else r'Vanilla Model'
    sns.lineplot(data=dict_train_grad_var[temp], x='epoch', y='Train Grad Var', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Average Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/evolutions/train_grad_var.png")

# Plot the val loss
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' if temp != 0 else r'Vanilla Model'
    sns.lineplot(data=dict_val_loss[temp], x='epoch', y='Val Loss', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Average Validation Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/evolutions/val_loss.png")

# Plot the FID
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' if temp != 0 else r'Vanilla Model'
    sns.lineplot(data=dict_val_fid[temp], x='epoch', y='FID_inf', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Average ' + r"$\overline{FID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/evolutions/fid.png")

# Plot the KID
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' if temp != 0 else r'Vanilla Model'
    sns.lineplot(data=dict_val_kid[temp], x='epoch', y='KID_inf', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{KID}_\infty$")
plt.title(f'Average ' + r"$\overline{KID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/evolutions/kid.png")

# Plot the MIFID
plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'$p=$' + f'{temp}' if temp != 0 else r'Vanilla Model'
    sns.lineplot(data=dict_val_mifid[temp], x='epoch', y='MIFID_inf', label=label)
plt.xlabel("Epoch")
plt.ylabel(r"$\overline{MIFID}_\infty$")
plt.title(f'Average ' + r"$\overline{MIFID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/evolutions/mifid.png")

# Plot boxplot of the final loss and metrics for each temperature
for temp in TEMPS:
    final_train_loss = dict_train_loss[temp].groupby('experiment').tail(5)
    final_train_grad_var = dict_train_grad_var[temp].groupby('experiment').tail(5)
    final_val_loss = dict_val_loss[temp].groupby('experiment').tail(5)
    final_fid = dict_val_fid[temp].groupby('experiment').tail(5)
    final_kid = dict_val_kid[temp].groupby('experiment').tail(5)
    final_mifid = dict_val_mifid[temp].groupby('experiment').tail(5)

    final_train_loss['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_train_grad_var['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_val_loss['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_fid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_kid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"
    final_mifid['temp'] = f"p={temp}" if temp != 0 else "Vanilla Model"

    if temp == 0:
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
    
plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_train_loss, x='Train Loss', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Final Train Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/boxplots/final_train_loss.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_train_grad_var, x='Train Grad Var', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title(f'Final Train Gradient Variance for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/boxplots/final_train_grad_var.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_val_loss, x='Val Loss', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$-\log(p(\mathbf{x}|\theta))$")
plt.title(f'Final Validation Loss for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/boxplots/final_val_loss.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_fid, x='FID_inf', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/boxplots/final_fid.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_kid, x='KID_inf', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\overline{KID}_\infty$")
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/boxplots/final_kid.png")

plt.figure(figsize=(15, 3))
sns.boxplot(data=all_final_mifid, x='MIFID_inf', y='temp', fill=False, orient='h', hue='temp')
plt.ylabel(r"$\overline{MIFID}_\infty$")
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + f' for {NUM_EXPERIMENTS} Experiments')
plt.savefig(f"results/boxplots/final_mifid.png")

# Plot final fid against variance, with error bars
fid_var = pd.merge(all_final_train_grad_var, all_final_fid, on=['experiment', 'temp'])

plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    mean_fid = fid_var[fid_var['temp'] == label]['FID_inf'].mean()
    std_fid = fid_var[fid_var['temp'] == label]['FID_inf'].std()
    mean_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = fid_var[fid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_fid, xerr=std_var, yerr=std_fid, capsize=2, marker='x', label=label)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{FID}_\infty$")
plt.title(f'Final ' + r"$\overline{FID}_\infty$" + ' against Validation Gradient Variance')
plt.legend()
plt.savefig(f"results/relationships/fid_var.png")

# Plot final mifid against variance, with error bars
mifid_var = pd.merge(all_final_train_grad_var, all_final_mifid, on=['experiment', 'temp'])

plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    mean_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].mean()
    std_mifid = mifid_var[mifid_var['temp'] == label]['MIFID_inf'].std()
    mean_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = mifid_var[mifid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_mifid, xerr=std_var, yerr=std_mifid, fmt='x', label=label, capsize=2)
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{MIFID}_\infty$")
plt.title(f'Final ' + r"$\overline{MIFID}_\infty$" + ' against Validation Gradient Variance')
plt.legend()
plt.savefig(f"results/relationships/mifid_var.png")

# Plot final kid against variance, with error bars
kid_var = pd.merge(all_final_train_grad_var, all_final_kid, on=['experiment', 'temp'])

plt.figure(figsize=(10, 6))
for temp in TEMPS:
    label = r'p=' + f'{temp}' if temp != 0 else r'Vanilla Model'
    mean_kid = kid_var[kid_var['temp'] == label]['KID_inf'].mean()
    std_kid = kid_var[kid_var['temp'] == label]['KID_inf'].std()
    mean_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].mean()
    std_var = kid_var[kid_var['temp'] == label]['Train Grad Var'].std()
    plt.errorbar(mean_var, mean_kid, xerr=std_var, yerr=std_kid, label=label, capsize=2, marker='x')
plt.xlabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.ylabel(r"$\overline{KID}_\infty$")
plt.title(f'Final ' + r"$\overline{KID}_\infty$" + ' against Validation Gradient Variance')
plt.legend()
plt.savefig(f"results/relationships/kid_var.png")