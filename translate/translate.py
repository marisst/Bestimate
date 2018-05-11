import sys

from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed, save_json

def get_sentences_for_key(datapoint, key):

    sentences = []
    if key == SUMMARY_FIELD_KEY or key == TOTAL_KEY:
        sentences = datapoint[SUMMARY_FIELD_KEY]

    if (key == DESCRIPTION_FIELD_KEY or key == TOTAL_KEY) and datapoint.get(DESCRIPTION_FIELD_KEY) is not None:
        sentences = sentences + datapoint.get(DESCRIPTION_FIELD_KEY)

    return sentences

def translate_text(text, dictionary):
    
    translated_words = []
    for word in text.split():
        if word in dictionary:
            if dictionary.get(word) is None:
                continue
            translated_words.append(dictionary[word])

    return " ".join([str(word) for word in translated_words])

def translate_data(dataset, key, dictionary, labeling):

    filename = get_dataset_filename(dataset, labeling, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    data = load_json(filename)

    print("Translating dataset")
    numeric_data = []
    for datapoint in data:
        numeric_datapoint = {}
        numeric_datapoint[NUMERIC_TEXT_KEY] = []
        sentences = get_sentences_for_key(datapoint, key)
        for sentence in sentences:
            numeric_datapoint[NUMERIC_TEXT_KEY].append(translate_text(sentence, dictionary))
        if TIMESPENT_FIELD_KEY in datapoint:
            numeric_datapoint[TIMESPENT_FIELD_KEY] = datapoint[TIMESPENT_FIELD_KEY]
        numeric_data.append(numeric_datapoint)

    filename = get_dataset_filename(dataset, labeling, NUMERIC_POSTFIX, JSON_FILE_EXTENSION)
    save_json(filename, numeric_data)
    print("Dataset translated and saved at", filename)

def translate_dataset(dataset, key):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)
    translate_data(dataset, key, dictionary, LABELED_FILENAME)
    translate_data(dataset, key, dictionary, UNLABELED_FILENAME)

if len(sys.argv) != 3:
    print("Please select one dataset and one of the following field keys:", SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TOTAL_KEY)
    sys.exit()

translate_dataset(sys.argv[1], sys.argv[2])