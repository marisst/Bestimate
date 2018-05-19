import numpy as np
from gensim.models import Word2Vec

from utilities.data_utils import get_issue_counts
from utilities.file_utils import load_json
from utilities.arrange import shuffle, split_train_test
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


def load_and_arrange(dataset, split_percentage, embeddings, max_words):

    data_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    filtered_data = load_json(data_filename)
    datapoint_count = len(filtered_data)
    shuffled_data = ordered_shuffle(filtered_data)

    if embeddings == "spacy":
        lookup_filename = get_dataset_filename(dataset, ALL_FILENAME, SPACY_LOOKUP_POSTFIX, JSON_FILE_EXTENSION)
        lookup = load_json(lookup_filename)
        embedding_size = len(next(iter(lookup.values())))

    if embeddings == "gensim":
        model_filename = get_dataset_filename(dataset, ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
        model = Word2Vec.load(model_filename)
        lookup = model.wv
        embedding_size = model.vector_size

    x = np.zeros((datapoint_count, max_words, embedding_size))
    for i, datapoint in enumerate(shuffled_data):
        text = merge_sentences(datapoint.get(SUMMARY_FIELD_KEY) + datapoint.get(DESCRIPTION_FIELD_KEY, []))
        words = text.split()
        words = [word for word in words if word in lookup][:max_words]
        start_index = max_words - len(words)
        for j, word in enumerate(words):
            x[i, start_index + j] = np.array(lookup[word])

    y = np.array([datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR for datapoint in shuffled_data])

    shuffled_data = shuffle((x, y))
    return split_train_test(shuffled_data, split_percentage)