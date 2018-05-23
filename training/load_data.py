import gc
import numpy as np
from gensim.models import Word2Vec

from utilities.data_utils import get_issue_counts
from utilities.file_utils import load_json
from utilities.constants import *
from utilities.string_utils import merge_sentences

def ordered_shuffle(data):

    np.random.seed(7)
    issue_counts = get_issue_counts(data)
    
    project_data = {}
    for project in issue_counts:
        project_id = project[0]
        project_data[project_id] = [datapoint for datapoint in data if datapoint[PROJECT_FIELD_KEY] == project_id]
        project_data[project_id] = sorted(project_data[project_id], key=lambda datapoint: datapoint[ID_FIELD_KEY], reverse=True)

    shuffled_data = []
    datapoint_count = len(data)
    for i in range(datapoint_count):
        project_ids = list(project_data.keys())
        probabilities = [len(project_data[project_id]) / (datapoint_count - i) for project_id in project_ids]
        project_id = np.random.choice(project_ids, None, p=probabilities)
        shuffled_data.append(project_data[project_id].pop())
        if len(project_data[project_id]) == 0:
            del project_data[project_id]

    return shuffled_data


def split(data, split_indices):
    return (data[:split_indices[0]], data[split_indices[0]:split_indices[1]], data[split_indices[1]:])

def split_train_test_val(data, split_percentages):

    x, y = data
    split_indices = len(y) * split_percentages[0] // 100, len(y) * (split_percentages[0] + split_percentages[1]) // 100

    x_train, x_test, x_valid = split(x, split_indices)
    y_train, y_test, y_valid = split(y, split_indices)
    y = None

    print("Data splitted in training and testing sets")

    return (x_train, y_train, x_test, y_test, x_valid, y_valid)


def load_and_arrange(dataset, split_percentage, embeddings, max_words, labeled_data=None, spacy_lookup=None, gensim_model=None):

    if labeled_data is None:
        data_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        labeled_data = load_json(data_filename)

    datapoint_count = len(labeled_data)
    shuffled_data = ordered_shuffle(labeled_data)

    if embeddings == "spacy":
        if spacy_lookup is None:
            lookup_filename = get_dataset_filename(dataset, ALL_FILENAME, SPACY_LOOKUP_POSTFIX, JSON_FILE_EXTENSION)
            lookup = load_json(lookup_filename)
        else:
            lookup = spacy_lookup
        embedding_size = len(next(iter(spacy_lookup.values())))

    if embeddings == "gensim":
        if gensim_model is None:
            model_filename = get_dataset_filename(dataset, ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
            gensim_model = Word2Vec.load(model_filename)
        lookup = gensim_model.wv
        embedding_size = gensim_model.vector_size

    x = np.zeros((datapoint_count, max_words, embedding_size))
    for i, datapoint in enumerate(shuffled_data):
        text = merge_sentences(datapoint.get(SUMMARY_FIELD_KEY) + datapoint.get(DESCRIPTION_FIELD_KEY, []))
        words = text.split()
        words = [word for word in words if word in lookup][:max_words]
        start_index = max_words - len(words)
        for j, word in enumerate(words):
            x[i, start_index + j] = np.array(lookup[word])

    y = np.array([datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR for datapoint in shuffled_data])

    shuffled_data = None
    gc.collect

    return split_train_test_val((x, y), split_percentage)