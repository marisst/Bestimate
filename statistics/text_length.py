import matplotlib.pyplot as plt
import numpy as np
import sys

from preprocessing import projects
from utilities import load_data
from utilities.constants import *

def show_histogram(dataset):

    filename = get_filtered_dataset_filename(dataset)
    data = load_data.load_json(filename)

    if data is None:
        return

    need_upper_limit = input("Would you like to put a constraint on the maximum text length displayed? (y/n) ") == "y"
    if need_upper_limit:
        upper_limit = int(input("Please enter the upper text length limit: "))

    bins = int(input("Please input the number of bins: "))

    text_lengths = [len(datapoint.get(SUMMARY_FIELD_KEY, "")) + len(datapoint.get(DESCRIPTION_FIELD_KEY, "")) for datapoint in data]

    min_length = min(text_lengths)
    max_length = max(text_lengths)
    if need_upper_limit:
        max_length = min(upper_limit, max_length)

    plt.figure(figsize=(12, 7))
    plt.hist(text_lengths, bins = bins, range = (min_length, max_length))
    plt.xticks(np.arange(0, max_length, (max_length - min_length) / bins))
    plt.xlim(min_length, max_length)

    load_data.create_folder_if_needed(STATISTICS_FOLDER)
    filename = get_statistics_image_filename(dataset, TEXT_LENGTH_STAT)
    plt.savefig(filename, bbox_inches=PLOT_BBOX_INCHES)

    print("Text length histogram saved at %s" % filename)

show_histogram(sys.argv[1])