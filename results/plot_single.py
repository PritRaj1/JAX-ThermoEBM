import matplotlib.pyplot as plt
import pandas as pd
import optax


df = pd.read_csv("logs/CelebA/p=0/batch=10/experiment0.csv")
df2 = pd.read_csv("logs/CelebA/p=1/batch=10/experiment0.csv")

# Plot df["Train Loss"]
plt.figure()
plt.plot(df["Train Grad Var"], label="p=0")
plt.plot(df2["Train Grad Var"], label="p=1")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss")
plt.savefig("results/1.png")

