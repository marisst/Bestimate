import sys

from pre.load_data import load_and_arange

# training parameters
split_percentage = 75
window_size = 20

def train_on_dataset(dataset):

    data = load_and_arange(dataset, window_size, split_percentage)

train_on_dataset(sys.argv[1])