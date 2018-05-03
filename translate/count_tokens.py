import json
import sys

from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed

TOTAL_KEY = "total"

def count_tokens(dataset):

    filename = get_filtered_dataset_filename(dataset)
    data = load_json(filename)

    token_counts = {
        SUMMARY_FIELD_KEY: {},
        DESCRIPTION_FIELD_KEY: {}, 
        TOTAL_KEY: {}
    }

    for datapoint in data:

        if datapoint.get(SUMMARY_FIELD_KEY) is not None:
            summary_words = datapoint[SUMMARY_FIELD_KEY].split()
            for word in summary_words:
                token_counts[SUMMARY_FIELD_KEY][word] = token_counts[SUMMARY_FIELD_KEY].get(word, 0) + 1
                token_counts[TOTAL_KEY][word] = token_counts[TOTAL_KEY].get(word, 0) + 1
        
        if datapoint.get(DESCRIPTION_FIELD_KEY) is not None:
            description_word = datapoint[DESCRIPTION_FIELD_KEY].split()
            for word in description_word:
                token_counts[DESCRIPTION_FIELD_KEY][word] = token_counts[DESCRIPTION_FIELD_KEY].get(word, 0) + 1
                token_counts[TOTAL_KEY][word] = token_counts[TOTAL_KEY].get(word, 0) + 1

    for key in token_counts:
        token_counts[key] = sorted(token_counts[key].items(), key=lambda x: x[1], reverse=True)

    create_folder_if_needed(TOKEN_COUNT_DATA_FOLDER)
    filename = get_token_count_filename(dataset)
    with open(filename, "w") as file:
        json.dump(token_counts, file)

if len(sys.argv) != 2:
    print("Please choose one dataset")
    sys.exit();

count_tokens(sys.argv[1])