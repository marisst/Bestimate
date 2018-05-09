import numpy as np

from utilities.constants import *
from utilities.load_data import load_json
from utilities.arrange import shuffle

def load_data(dataset, max_sentence_length):

    filename = get_dataset_filename(dataset, LABELED_FILENAME, NUMERIC_POSTFIX, JSON_FILE_EXTENSION)
    data = load_json(filename)

    x_arr = [[int(word) for word in datapoint[NUMERIC_TEXT_KEY].split()] for datapoint in data]
    y_arr = [datapoint[TIMESPENT_FIELD_KEY] for datapoint in data]

    x = np.zeros((len(x_arr), max_sentence_length), dtype='int32')
    for i, sentence in enumerate(x_arr):
        start_index = max([max_sentence_length - len(sentence), 0])
        for j in range(min([len(sentence), max_sentence_length])):
            x[i, start_index + j] = sentence[j]
    y = np.array(y_arr)

    return shuffle((x, y))

