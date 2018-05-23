from utilities.constants import *
from utilities.file_utils import load_json, create_folder_if_needed, save_json
from utilities.string_utils import merge_sentences

def count_tokens(dataset, notes_filename, data=None, save=True):

    if data is None:

        labeled_data_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        labeled_data = load_json(labeled_data_filename)

        unlabeled_data_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        unlabeled_data = load_json(unlabeled_data_filename)

        data = labeled_data
        if unlabeled_data is not None:
            data = data + unlabeled_data

    print("Counting tokens...")

    token_counts = {}

    for datapoint in data:

        for text_key in [SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY]:
            if datapoint.get(text_key) is not None:
                summary_words = merge_sentences(datapoint[text_key]).split()
                for word in summary_words:
                    token_counts[word] = token_counts.get(word, 0) + 1

    print("Sorting...")
    token_counts = sorted(token_counts.items(), key=lambda x: x[1], reverse=True)

    if save == True:
        filename = get_dataset_filename(dataset, ALL_FILENAME, TOKEN_COUNT_POSTFIX, JSON_FILE_EXTENSION)
        save_json(filename, token_counts)
        print("Token counts and frequencies saved at %s" % filename)

    with open(notes_filename, "a") as notes_file:
        print("%d different unique tokens" % (len(token_counts)), file=notes_file)

    return token_counts