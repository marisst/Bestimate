import sys

from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed, save_json

def get_text_for_key(datapoint, key):

    text = ""
    if key == SUMMARY_FIELD_KEY or key == TOTAL_KEY:
        text = datapoint[SUMMARY_FIELD_KEY]

    if (key == DESCRIPTION_FIELD_KEY or key == TOTAL_KEY) and datapoint.get(DESCRIPTION_FIELD_KEY) is not None:
        separator_if_needed = " " if len(text) > 0 else ""
        text = "%s%s%s" % (text, separator_if_needed, datapoint[DESCRIPTION_FIELD_KEY])

    return text

def translate_text(text, dictionary):
    
    translated_words = []
    for word in text.split():
        if word in dictionary:
            translated_words.append(dictionary[word])

    return translated_words

def translate_dataset(dataset, key, enable_prelearning):

    dictionary_filename = get_dictionary_filename(dataset)
    dictionary = load_json(dictionary_filename)

    data_filename = get_filtered_dataset_filename(dataset)
    data = load_json(data_filename)

    print("Translating dataset")
    numeric_data = []
    for datapoint in data:
        text = get_text_for_key(datapoint, key)
        numeric_text = translate_text(text, dictionary)
        
        numeric_datapoint = {}
        numeric_datapoint[NUMERIC_TEXT_KEY] = numeric_text
        if TIMESPENT_FIELD_KEY in datapoint:
            numeric_datapoint[TIMESPENT_FIELD_KEY] = datapoint[TIMESPENT_FIELD_KEY]
        numeric_data.append(numeric_datapoint)

    create_folder_if_needed(NUMERIC_FOLDER)
    filename = get_numeric_folder(dataset)
    save_json(filename, numeric_data)

    print("Dataset translated and saved at", filename)

if len(sys.argv) != 3:
    print("Please select one dataset and one of the following field keys:", SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TOTAL_KEY)
    sys.exit();

enable_prelearning = input("Would you like to enable prelearning? (y/n) ") == "y"

translate_dataset(sys.argv[1], sys.argv[2], enable_prelearning)