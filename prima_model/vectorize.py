import numpy as np
import pickle
import spacy
import sys

from utilities import load_data, string_utils
from utilities.constants import *

nlp = spacy.load('en')

def get_vectors(text, max_text_length, embedding_size):

    vectors = np.zeros((max_text_length, embedding_size))
    if text != None:
        doc = nlp(text)
        j = max(0, max_text_length - len(doc))  
        for token in doc:
            vectors[j] = np.array(token.vector)
            j=j+1
            if j==max_text_length:
                break
    return vectors

def serialize(vectorized_data, dataset_name):

    load_data.create_folder_if_needed(VECTORIZED_DATA_FOLDER)
    filename = get_vectorized_dataset_filename(dataset_name)

    with open(filename, "wb") as file:
        pickle.dump(vectorized_data, file, PICKLE_PROTOCOL)

def vectorize_dataset(dataset):

    filename = get_filtered_dataset_filename(dataset)
    filtered_data = load_data.load_json(filename)
    datapoint_count = len(filtered_data)
    max_text_length = int(input("Please input maximum text length: "))

    x = np.zeros((datapoint_count, max_text_length, SPACY_EMBEDDING_SIZE))
    y = np.zeros((datapoint_count))

    for i, datapoint in enumerate(filtered_data):

        x[i] = get_vectors("%s %s" % (datapoint.get(SUMMARY_FIELD_KEY, ""), datapoint.get(DESCRIPTION_FIELD_KEY, "")),
            max_text_length, SPACY_EMBEDDING_SIZE)
        y[i] = datapoint[TIMESPENT_FIELD_KEY]

        print("%d (%.2f%%) of %d records processed" % string_utils.get_part_strings(i + 1, datapoint_count))

    serialize((x, y), dataset)

vectorize_dataset(sys.argv[1])