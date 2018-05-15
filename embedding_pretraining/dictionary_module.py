from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed, save_json
from utilities.string_utils import get_part_strings

import sys

def create_dictionary(dataset, field_key, minimum_repetitions):

    filename = get_dataset_filename(dataset, ALL_FILENAME, TOKEN_COUNT_POSTFIX, JSON_FILE_EXTENSION)
    token_counts = load_json(filename)

    dictionary = {}
    for i, word in enumerate(token_counts[field_key]):

        if token_counts[field_key][word]["count"] < minimum_repetitions:
            continue
            
        dictionary[word] = i+1

    filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    save_json(filename, dictionary)

    print("%d (%.0f%%) of %d words included in dictionary" % get_part_strings(len(dictionary), len(token_counts[field_key])))
    print("Dictionary created and saved at %s" % filename)