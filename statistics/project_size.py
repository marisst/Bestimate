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

    need_upper_limit = input("Would you like to put a constraint on the maximum project size displayed? (y/n) ") == "y"
    if need_upper_limit:
        upper_limit = int(input("Please enter the upper project size limit: "))

    bins = int(input("Please input the number of bins: "))

    project_issue_counts = projects.get_issue_counts(data)
    issue_counts = [c[1] for c in project_issue_counts]
    
    min_size = min(issue_counts)
    max_size = max(issue_counts)
    if need_upper_limit:
        max_size = min(upper_limit, max_size)

    plt.figure(figsize=(12, 7))
    plt.hist(issue_counts, bins = bins, range = (min_size, max_size))
    plt.xticks(np.arange(0, max_size, (max_size - min_size) / bins))
    plt.xlim(min_size, max_size)

    load_data.create_folder_if_needed(STATISTICS_FOLDER)
    filename = get_statistics_image_filename(dataset, PROJECT_SIZE_STAT)
    plt.savefig(filename, bbox_inches=PLOT_BBOX_INCHES)

    print("Project size histogram saved at %s" % filename)

show_histogram(sys.argv[1])