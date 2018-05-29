import gc
from keras.preprocessing.sequence import pad_sequences
import numpy as np
np.set_printoptions(threshold=np.nan)

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

    x_train, x_test, x_valid = [], [], []

    for input_data in x:
        t_train, t_test, t_valid = split(input_data, split_indices)
        x_train.append(t_train)
        x_test.append(t_test)
        x_valid.append(t_valid)


    y_train, y_test, y_valid = split(y, split_indices)
    y = None

    print("Data splitted in training and testing sets")
    return (x_train, y_train, x_test, y_test, x_valid, y_valid)


def encrypt_text(text, max_length, lookup, string_dictionary, vector_dictionary):

    encrypted_text = []
    words = text.split()
    j = 0
    for word in words:
        
        word_vector = lookup(word)
        if word_vector is None:
            continue

        if word in string_dictionary:
            encrypted_word = string_dictionary[word]
        else:
            encrypted_word = len(string_dictionary) + 1
            string_dictionary[word] = encrypted_word
            vector_dictionary.append(word_vector)


        encrypted_text.append(encrypted_word)
        j += 1
        if j >= max_length:
            break

    return encrypted_text, string_dictionary, vector_dictionary
    

def load_and_arrange(dataset, split_percentage, max_words, lookup, labeled_data=None):

    if labeled_data is None:
        data_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        labeled_data = load_json(data_filename)

    shuffled_data = ordered_shuffle(labeled_data)
    del labeled_data

    x_strings = [(merge_sentences(datapoint.get(SUMMARY_FIELD_KEY)), merge_sentences(datapoint.get(DESCRIPTION_FIELD_KEY, []))) for datapoint in shuffled_data]
    y = np.array([datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR for datapoint in shuffled_data])
    del shuffled_data

    print("Converting data to numeric format and creating vector dictionary...")
    x_sentences = [[], []]
    string_dictionary = {}
    vector_dictionary = []

    for summary, description in x_strings:

        encrypted_summary, string_dictionary, vector_dictionary = encrypt_text(summary, max_words[0], lookup, string_dictionary, vector_dictionary)
        encrypted_description, string_dictionary, vector_dictionary = encrypt_text(description, max_words[1], lookup, string_dictionary, vector_dictionary)
        
        x_sentences[0].append(encrypted_summary)
        x_sentences[1].append(encrypted_description)

    vector_dictionary.insert(0, [0] * len(vector_dictionary[0]))
    vector_dictionary = np.array(vector_dictionary)
    x = [pad_sequences(sentences, maxlen=max_words[i]) for i, sentences in enumerate(x_sentences)]

    return split_train_test_val((x, y), split_percentage), vector_dictionary