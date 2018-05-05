from utilities import load_data
from utilities.arrange import shuffle, split_train_test
from utilities.constants import *

def load_and_arrange(dataset, split_percentage):

    filename = get_vectorized_dataset_filename(dataset)
    data = load_data.load_pickle(filename)
    shuffled_data = shuffle(data)
    return split_train_test(shuffled_data, split_percentage)