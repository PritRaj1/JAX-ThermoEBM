import matplotlib.pyplot as plt
import pandas as pd
import optax


df = pd.read_csv("logs/CelebA/p=0/batch=100/experiment0.csv")

# Plot df["Train Loss"]
plt.figure()
plt.plot(df["Train Loss"], label="Val")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss")
plt.savefig("results/1.png")

