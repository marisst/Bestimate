import matplotlib.pyplot as plt
from utilities import load_data
from utilities.constants import *
import sys
import numpy as np

def show_histogram(dataset):

    filename = get_vectorized_dataset_filename(dataset)
    x, y = load_data.load_pickle(filename)

    if y is None:
        return 

    max_hours = int(input("Please input the maximum number of hours to display in the histogram: "))

    plt.figure(figsize=(12, 7))
    plt.hist(y / SECONDS_IN_HOUR, bins = max_hours * 12, range = (0, max_hours - 1 / SECONDS_IN_HOUR))
    plt.xticks(np.arange(0, max_hours + 1, 1))
    plt.xlim(0, max_hours)

    load_data.create_folder_if_needed(STATISTICS_FOLDER)
    filename = get_statistics_image_filename(dataset, LABEL_DISTRIBUTION_STAT)
    plt.savefig(filename, bbox_inches=PLOT_BBOX_INCHES)

    print("Label distribution histogram saved at %s" % filename)

show_histogram(sys.argv[1])