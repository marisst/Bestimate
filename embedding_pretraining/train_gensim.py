from gensim.models import Word2Vec
import sys

from utilities.file_utils import load_json, create_folder_if_needed
from utilities.constants import *


def train_gensim(dataset, algorithm, embedding_size, minimum_count, window_size, iterations, notes_filename):

    labeled_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    unlabeled_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)

    labeled_data = load_json(labeled_filename)
    unlabeled_data = load_json(unlabeled_filename)

    data = labeled_data if labeled_data is not None else [] + unlabeled_data if unlabeled_data is not None else []

    training_sentences = []
    for datapoint in data:
        sentences = datapoint[SUMMARY_FIELD_KEY]
        if datapoint.get(DESCRIPTION_FIELD_KEY) is not None:
            sentences = sentences + datapoint.get(DESCRIPTION_FIELD_KEY)
        for sentence in sentences:
            training_sentences.append([word for word in sentence.split()])
    print("Sentences prepared")

    model = Word2Vec(training_sentences,
    min_count=minimum_count,
    size=embedding_size,
    window=window_size,
    sg=1 if algorithm == "skip-gram" else 0,
    compute_loss=True,
    iter=iterations)

    with open(notes_filename, "a") as notes_filename:
        print("Gensim model loss:", model.get_latest_training_loss(), file=notes_filename)

    filename = get_dataset_filename(dataset, ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
    model.save(filename)
    print("Model saved at", filename)