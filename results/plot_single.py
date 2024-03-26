import matplotlib.pyplot as plt
import pandas as pd
import optax


df = pd.read_csv("logs/CelebA/p=0/batch=100/experiment0.csv")

# Plot df["Train Loss"]
plt.figure()
plt.plot(df["Train Loss"][0:], label="Train Loss")
plt.plot(df["Val Loss"][0:], label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Train Loss")
plt.savefig("results/1.png")

LR_schedule = optax.exponential_decay(0.0002, 750, 0.998, 1500, end_value=0.00002)
# Plot LR_schedule
plt.figure()
plt.plot([LR_schedule(i) for i in range(18000)])
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Parameter update")
plt.savefig("results/2.png")

