import numpy as np

from utilities.constants import *
from utilities.load_data import load_json

def get_prepoint_params(sentences, window_size):

    datapoint_params = {}
    index = 0
    for i, sentence in enumerate(sentences):

        if len(sentence) < window_size:
            continue

        last_start_index = len(sentence) - window_size
        for j in range(last_start_index):
            datapoint_params[index] = (i, j)
            index = index + 1

    print("Generated %d pretraining datapoints" % len(datapoint_params))
    return datapoint_params

def get_sentences(dataset):

    labeled_filename = get_dataset_filename(dataset, LABELED_FILENAME, NUMERIC_POSTFIX, JSON_FILE_EXTENSION)
    labeled_data = load_json(labeled_filename)
    
    unlabeled_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, NUMERIC_POSTFIX, JSON_FILE_EXTENSION)
    unlabeled_data = load_json(unlabeled_filename)

    data = labeled_data + unlabeled_data
    sentences = []
    for datapoint in data:
        for sentence in datapoint[NUMERIC_TEXT_KEY]:
            sentences.append([int(word) for word in sentence.split()])
    return sentences

def get_vocabulary_size(dataset):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)
    return len(dictionary) + 1




