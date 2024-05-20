import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np

# Set plot styling
sns.set(font_scale=2)
sns.set_style(
    "whitegrid",
    rc={"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern"]},
)

Temps = ['Vanilla Model,\n' + r"$N_{batch}=75, K_{\mathbf{z}|\mathbf{x}}=20$", r'$N_t=10, K_{\mathbf{z}|\mathbf{x}}=20$', r'$N_t=10, K_{\mathbf{z}|\mathbf{x}}=40$', r'$N_t=30, K_{\mathbf{z}|\mathbf{x}}=20$']
FLOPS = [159780470784.0, 266255302656.0, 266266116096.0, 266255302656.0]
TIMES = [3205.51, 28261.13, 53126.38, 79707.93]

# Plot bar chart with sns
plt.figure(figsize=(15, 11))
ax = sns.barplot(x=Temps, y=FLOPS, palette="magma")
ax.set_ylabel(r"FLOPS (flop/s)")
ax.set_yscale("log")

# Add text labels to the bars to indicate the value
for i in range(len(FLOPS)):
    ax.text(i, FLOPS[i], "{:.2e}".format(FLOPS[i]), ha='center', va='bottom')

plt.xticks(rotation=0)
plt.title(r"FLOPS to Evaluate $-\log(p(\mathbf{x}|\theta))$")
plt.tight_layout()
plt.savefig("results/flops.png")



plt.figure(figsize=(15, 10))
ax = sns.barplot(x=Temps, y=TIMES, palette="viridis")
ax.set_ylabel(r"Time (s)")
ax.set_yscale("log")

# Add text labels to the bars to indicate the value
for i in range(len(TIMES)):
    ax.text(i, TIMES[i], "{:.2e}".format(TIMES[i]), ha='center', va='bottom')

plt.xticks(rotation=0)
plt.title(r"Time Taken to Train for 50 Epochs")
plt.tight_layout()
plt.savefig("results/times.png")