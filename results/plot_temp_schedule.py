import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plot styling
sns.set(font_scale=1.2)
sns.set_style("whitegrid", rc ={'text.usetex' : True, 'font.family' : 'serif', 'font.serif' : ['Computer Modern']})


p_list = [0, 1, 3, 5, 7]
num_temps = 100

plt.figure()
for p in p_list:
    temp = np.linspace(0, 1, num_temps) ** p
    label = "p = {}".format(p) if p != 0 else "p = 0 (Vanilla Model)"
    plt.plot(temp, label=label)
plt.xlabel("Schedule Index")
plt.ylabel("Temperature")
plt.title("Temperature Schedule")
plt.legend(loc="upper left")
plt.savefig("results/temperature_schedule.png")

