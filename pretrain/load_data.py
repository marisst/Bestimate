import numpy as np

from utilities.constants import *
from utilities.load_data import load_json
from utilities.arrange import shuffle, split_train_test

def generate_prepoints(sentences, window_size):

    prepoints_x = []
    prepoints_y = []
    for sentence in sentences:

        if len(sentence) < window_size:
            continue

        last_start_index = len(sentence) - window_size
        for start_index in range(0, last_start_index):
            prepoints_x.append(sentence[start_index:start_index+window_size])
            prepoints_y.append(sentence[start_index+window_size])

    print("Generated %s pretraining datapoints" % len(prepoints_x))
    
    return np.array(prepoints_x), np.array(prepoints_y)


def load_and_arange(dataset, window_size, split_percentage):

    labeled_filename = get_dataset_filename(dataset, LABELED_FILENAME, NUMERIC_POSTFIX, JSON_FILE_EXTENSION)
    labeled_data = load_json(labeled_filename)
    
    unlabeled_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, NUMERIC_POSTFIX, JSON_FILE_EXTENSION)
    unlabeled_data = load_json(unlabeled_filename)

    data = labeled_data + unlabeled_data
    sentences = [[int(word) for word in datapoint[NUMERIC_TEXT_KEY].split()] for datapoint in data]

    prepoints = generate_prepoints(sentences, window_size)

    shuffled_prepoints = shuffle(prepoints)
    return split_train_test(shuffled_prepoints, split_percentage)



    


