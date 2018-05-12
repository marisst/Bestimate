import numpy as np
from gensim.models import Word2Vec

from utilities import load_data
from utilities.arrange import shuffle, split_train_test
from utilities.constants import *
from utilities.string_utils import merge_sentences

MAX_WORDS = 100

def load_and_arrange(dataset, split_percentage, embeddings):

    data_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    filtered_data = load_data.load_json(data_filename)
    datapoint_count = len(filtered_data)

    if embeddings == "spacy":
        lookup_filename = get_dataset_filename(dataset, ALL_FILENAME, SPACY_LOOKUP_POSTFIX, JSON_FILE_EXTENSION)
        lookup = load_data.load_json(lookup_filename)
        embedding_size = len(next(iter(lookup.values())))

    if embeddings == "gensim":
        model_filename = get_dataset_filename(dataset, ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
        model = Word2Vec.load(model_filename)
        lookup = model.wv
        embedding_size = model.vector_size

    x = np.zeros((datapoint_count, MAX_WORDS, embedding_size))
    for i, datapoint in enumerate(filtered_data):
        text = merge_sentences(datapoint.get(SUMMARY_FIELD_KEY) + datapoint.get(DESCRIPTION_FIELD_KEY, []))
        words = text.split()
        words = [word for word in words if word in lookup][:MAX_WORDS]
        start_index = MAX_WORDS - len(words)
        for j, word in enumerate(words):
            x[i, start_index + j] = np.array(lookup[word])

    y = np.array([datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR for datapoint in filtered_data])

    shuffled_data = shuffle((x, y))
    return split_train_test(shuffled_data, split_percentage)