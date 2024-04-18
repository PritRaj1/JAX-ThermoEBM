import matplotlib.pyplot as plt
import pandas as pd
import optax


df = pd.read_csv("logs/SVHN/p=0/batch=75/experiment0.csv")
# df2 = pd.read_csv("logs/SVHN/p=1/batch=75/experiment0.csv")
# df3 = pd.read_csv("logs/SVHN/p=3/batch=75/experiment0.csv")

# Plot df["Train Loss"]
plt.figure(figsize=(10, 6))
plt.plot(df["Train Loss"], label="p=0")
# plt.plot(df2["KID_inf"], label="p=1")
# plt.plot(df3["KID_inf"], label="p=3")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel(r"KID")
plt.title("KID_inf")
plt.savefig("results/1.png")

# $\mathrm{Var}_\theta\left[\nabla_\theta \right]$