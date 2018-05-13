from gensim.models import Word2Vec
import sys

from utilities.load_data import load_json, create_folder_if_needed
from utilities.constants import *

#https://www.shanelynn.ie/word-embeddings-in-python-with-spacy-and-gensim/
#https://radimrehurek.com/gensim/models/word2vec.html

def train_on_dataset(dataset):

    labeled_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    unlabeled_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    data = load_json(unlabeled_filename) + load_json(labeled_filename)

    training_sentences = []
    for datapoint in data:
        sentences = datapoint[SUMMARY_FIELD_KEY]
        if datapoint.get(DESCRIPTION_FIELD_KEY) is not None:
            sentences = sentences + datapoint.get(DESCRIPTION_FIELD_KEY)
        for sentence in sentences:
            training_sentences.append([word for word in sentence.split()])
    print("Sentences prepared")

    model = Word2Vec(training_sentences, size=10, window=5, sg=1, compute_loss=True, iter=5)
    print("Model computed, loss:", model.get_latest_training_loss())

    filename = get_dataset_filename(dataset, ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
    model.save(filename)
    print("Model saved at", filename)

train_on_dataset(sys.argv[1])