import sys
import numpy as np

from training import load_data as load
from training.data_generator import DataGenerator

def fake_lookup(word):
    return [1, 2, 3]

def calculate_diffs(training_dataset_id):

    data, vector_dictionary = load.load_and_arrange(
    training_dataset_id,
    (60, 20),
    False,
    (5,5),
    fake_lookup)
    _, y_train, _, _, _, y_valid = data

    training_median = np.median(y_train)
    validation_generator = DataGenerator((np.zeros((len(y_valid), 5)), np.zeros((len(y_valid), 5))), y_valid, 512, False, (1,1), vector_dictionary)

    selected_validation_y = np.zeros((validation_generator.__len__() * 512))
    for batch_i in range(validation_generator.__len__()):
        _, selected_validation_y[batch_i*512:] = validation_generator.__getitem__(batch_i)

    print(len(selected_validation_y))


if __name__ == "__main__":
    calculate_diffs(sys.argv[1])
