import sys; sys.path.append('..')
import torch

from src.utils.helper_functions import parse_input_file, get_data

config = parse_input_file('hyperparams.input')

dataset, val_dataset, config['IMAGE_DIM'] = get_data(config['DATASET'])

print(config)

# Take a subset of the dataset
train_data = torch.utils.data.Subset(dataset, config['NUM_TRAIN_DATA'])
val_data = torch.utils.data.Subset(val_dataset, config['NUM_VAL_DATA'])