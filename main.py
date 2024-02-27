import sys; sys.path.append('..')

from src.utils.helper_functions import parse_input_file, get_data

config = parse_input_file('hyperparams.input')

dataset, val_dataset, config['IMAGE_DIM'] = get_data(config['DATASET'])

print(config)
