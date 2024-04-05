import matplotlib.pyplot as plt
import pandas as pd
import optax


df = pd.read_csv("logs/CelebA/p=0/batch=75/experiment0.csv")
df2 = pd.read_csv("logs/CelebA/p=1/batch=75/experiment0.csv")

# Plot df["Train Loss"]
plt.figure(figsize=(10, 6))
plt.plot(df["Val Loss"], label="p=0")
plt.plot(df2["Val Loss"], label="p=1")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel(r"$\mathrm{Var}_\theta\left[\nabla_\theta \log(p(\mathbf{x}|\theta))\right]$")
plt.title("Train Grad Var")
plt.savefig("results/1.png")

