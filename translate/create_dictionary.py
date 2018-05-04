from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed, save_json

import sys

def create_dictionary(dataset, field_key):

    filename = get_token_count_filename(dataset)
    data = load_json(filename)

    dictionary = {}
    for i, word in enumerate(data[field_key]):
        dictionary[word] = i+1

    create_folder_if_needed(DICTIONARY_DATA_FOLDER)
    filename = get_dictionary_filename(dataset)
    save_json(filename, dictionary)

    print("Dictionary created and saved at %s" % filename)

if len(sys.argv) != 3:
    print("Please select one dataset and one of the following field keys:", SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TOTAL_KEY)
    sys.exit();

create_dictionary(sys.argv[1], sys.argv[2])