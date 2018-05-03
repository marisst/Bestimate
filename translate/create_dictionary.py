from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed

import sys

def create_dictionary(dataset):

    filename = get_token_count_filename(dataset)
    data = load_json(filename)

    dictionary = {}
    for i, word in enumerate(data[TOTAL_KEY]):
        dictionary[word] = i+1

    


if len(sys.argv) != 2:
    print("Please select one dataset")
    sys.exit();

create_dictionary(sys.argv[1])