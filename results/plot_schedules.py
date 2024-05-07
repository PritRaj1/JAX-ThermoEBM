import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import configparser
import optax

parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

# Set plot styling
sns.set(font_scale=1.1)
sns.set_style("whitegrid", rc ={'text.usetex' : True, 'font.family' : 'serif', 'font.serif' : ['Computer Modern']})

begin = int(parser["LR_SCHEDULE"]["BEGIN_EPOCH"]) * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])
LR_0 = float(parser["OPTIMIZER"]["G_INITIAL_LR"])
LR_1 = float(parser["OPTIMIZER"]["G_FINAL_LR"])
GAMMA = float(parser["LR_SCHEDULE"]["DECAY_RATE"])
STEP = int(parser["LR_SCHEDULE"]["STEP_INTERVAL"]) * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])

LR_schedule = optax.exponential_decay(init_value=LR_0, transition_steps=STEP, decay_rate=GAMMA, transition_begin=begin, end_value=LR_1)

epochs = np.linspace(0,int(parser["PIPELINE"]["NUM_EPOCHS"]))
steps = epochs * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])

plt.figure()
plt.plot(epochs,[LR_schedule(step) for step in steps])
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Parameter update")
plt.savefig("results/lr_schedule.png")

p_list = [0, 0.3, 0.1, 1, 2, 3, 6, 10]
num_temps = 100

plt.figure(figsize=(10, 6))
for p in p_list:
    temp = np.linspace(0, 1, num_temps) ** p
    label = "p = {}".format(p)
    plt.plot(temp, label=label)
plt.xlabel("Schedule Index")
plt.ylabel("Temperature")
plt.title("Temperature Schedule")
plt.legend(loc="upper left")
plt.savefig("results/temperature_schedule.png")

print(LR_schedule(steps[-1]))

