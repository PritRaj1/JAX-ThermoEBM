import optax
import matplotlib.pyplot as plt
import numpy as np
import configparser
parser = configparser.ConfigParser()
parser.read("hyperparams.ini")

begin = int(parser["LR_SCHEDULE"]["BEGIN_EPOCH"]) * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])
LR_0 = float(parser["OPTIMIZER"]["E_INITIAL_LR"])
LR_1 = float(parser["OPTIMIZER"]["E_FINAL_LR"])
GAMMA = float(parser["LR_SCHEDULE"]["DECAY_RATE"])
STEP = int(parser["LR_SCHEDULE"]["STEP_INTERVAL"]) * int(parser["PIPELINE"]["NUM_TRAIN_DATA"])/int(parser["PIPELINE"]["BATCH_SIZE"])

LR_schedule = optax.exponential_decay(init_value=LR_0, transition_steps=STEP, decay_rate=GAMMA, transition_begin=begin, end_value=LR_1)

epochs = np.linspace(0,120)
steps = epochs * 90

plt.figure()
plt.plot(epochs,[LR_schedule(step) for step in steps])
plt.xlabel("Epoch")
plt.ylabel("Learning Rate")
plt.title("Parameter update")
plt.savefig("results/plot_lr_schedule.png")
