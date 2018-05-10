from utilities.constants import *
from utilities.load_data import load_json, create_folder_if_needed, save_json
from utilities.string_utils import merge_sentences

def calculate_frequencies(token_counts):

    print("Calculating frequencies...")
    token_stats = {}
    for key in token_counts:
        token_stats[key] = {}
        word_count = sum([e[1] for e in token_counts[key]])
        for token_count in token_counts[key]:
            token_stats[key][token_count[0]] = {
                "count" : token_count[1],
                "frequency" : token_count[1] / word_count * 100
            }

    return token_stats

def count_tokens(dataset):

    labeled_data_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    labeled_data = load_json(labeled_data_filename)

    unlabeled_data_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    unlabeled_data = load_json(unlabeled_data_filename)

    data = labeled_data + unlabeled_data

    print("Counting tokens...")

    token_counts = {
        SUMMARY_FIELD_KEY: {},
        DESCRIPTION_FIELD_KEY: {}, 
        TOTAL_KEY: {}
    }

    for datapoint in data:

        if datapoint.get(SUMMARY_FIELD_KEY) is not None:
            summary_words = merge_sentences(datapoint[SUMMARY_FIELD_KEY]).split()
            for word in summary_words:
                token_counts[SUMMARY_FIELD_KEY][word] = token_counts[SUMMARY_FIELD_KEY].get(word, 0) + 1
                token_counts[TOTAL_KEY][word] = token_counts[TOTAL_KEY].get(word, 0) + 1
        
        if datapoint.get(DESCRIPTION_FIELD_KEY) is not None:
            description_word = merge_sentences(datapoint[DESCRIPTION_FIELD_KEY]).split()
            for word in description_word:
                token_counts[DESCRIPTION_FIELD_KEY][word] = token_counts[DESCRIPTION_FIELD_KEY].get(word, 0) + 1
                token_counts[TOTAL_KEY][word] = token_counts[TOTAL_KEY].get(word, 0) + 1

    print("Sorting...")
    for key in token_counts:
        token_counts[key] = sorted(token_counts[key].items(), key=lambda x: x[1], reverse=True)

    token_stats = calculate_frequencies(token_counts)

    filename = get_dataset_filename(dataset, ALL_FILENAME, TOKEN_COUNT_POSTFIX, JSON_FILE_EXTENSION)
    save_json(filename, token_stats)

    print("Token counts and frequencies saved at %s" % filename)