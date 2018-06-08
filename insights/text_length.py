import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sys

from utilities.file_utils import load_json, create_folder_if_needed
from utilities.constants import *
from utilities.string_utils import merge_sentences

def get_texts(data, field):

    if field == None:
        return [merge_sentences(datapoint.get(SUMMARY_FIELD_KEY) + datapoint.get(DESCRIPTION_FIELD_KEY, [])) for datapoint in data]
    
    if field == SUMMARY_FIELD_KEY:
        return [merge_sentences(datapoint.get(SUMMARY_FIELD_KEY)) for datapoint in data]

    if field == DESCRIPTION_FIELD_KEY:
        return [merge_sentences(datapoint.get(DESCRIPTION_FIELD_KEY, [])) for datapoint in data]

    print("Field not recognized")
    sys.exit()


def get_x_label(field):

    if field == None:
        return "Text length, words"

    if field == SUMMARY_FIELD_KEY:
        return "Summary text length, words"

    if field == DESCRIPTION_FIELD_KEY:
        return "Description text length, words"

    print("Field not recognized")
    sys.exit()


def show_histogram(dataset, labeling = LABELED_FILENAME, field = None):

    if labeling == ALL_FILENAME:
        labeled_filename = get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        unlabeled_filename = get_dataset_filename(dataset, UNLABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        data = load_json(labeled_filename) + load_json(unlabeled_filename)
    else:
        filename = get_dataset_filename(dataset, labeling, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
        data = load_json(filename)

    if data is None:
        print("No data was selected")
        sys.exit()

    texts = get_texts(data, field)
    text_lengths = [len(text.split()) for text in texts]

    print("Mean, words:", np.mean(text_lengths))
    print("Median, words:", np.median(text_lengths))
    print("Standard deviation, words:", np.std(text_lengths))
    print("Minimum, words:", np.min(text_lengths))
    print("Maximum, words:", np.max(text_lengths))

    need_upper_limit = input("Would you like to put a constraint on the maximum text length displayed? (y/n) ") == "y"
    if need_upper_limit:
        upper_limit = int(input("Please enter the upper text length limit (words): "))

    min_length = min(text_lengths)
    max_length = max(text_lengths)
    if need_upper_limit:
        max_length = min(upper_limit, max_length)

    suggested_number_of_bins = [i for i in range(5, 30) if (max_length - min_length) % i == 0]
    print("Suggested number of bins is:", *suggested_number_of_bins)
    bins = int(input("Please input the number of bins: "))

    plt.figure(figsize=(12, 7))
    plt.hist(text_lengths, bins = bins, range = (min_length, max_length))
    step = (max_length - min_length) / bins
    plt.xticks(np.arange(min_length, max_length + 1, step))
    plt.xlim(min_length, max_length)
    plt.xlabel(get_x_label(field))
    plt.ylabel("Number of records")

    create_folder_if_needed(STATISTICS_FOLDER)
    filename = get_statistics_image_filename(dataset, TEXT_LENGTH_STAT)
    plt.savefig(filename, bbox_inches=PLOT_BBOX_INCHES)

    print("Text length histogram saved at %s" % filename)

show_histogram(
    sys.argv[1],
    sys.argv[2] if len(sys.argv) > 2 else LABELED_FILENAME,
    sys.argv[3] if len(sys.argv) > 3 else None)