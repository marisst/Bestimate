from enum import Enum
import json
import sys

from preprocess import projects
from utilities import load_data, string_utils
from utilities.constants import *

class Extreme(Enum):
    MINIMUM = min
    MAXIMUM = max

def load_dataset(dataset, labeling):
    
    filename = get_dataset_filename(dataset, labeling, MERGED_POSTFIX, JSON_FILE_EXTENSION)
    return load_data.load_json(filename)

def remove_unlabeled_datapoints(data):

    labeled_data = [datapoint for datapoint in data if TIMESPENT_FIELD_KEY in datapoint]
    if(len(labeled_data) != len(data)):
        print("%d (%d%%) of %d datapoints were removed because they were unlabeled" % string_utils.get_part_strings(len(data)-len(labeled_data), len(data)))
    return labeled_data

def print_extreme(data, extreme):

    extreme_datapoint = extreme.value(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])
    print("The %s timespent is %.2f hours for the following issue: %s"
        % (
            extreme.name,
            extreme_datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR,
            extreme_datapoint[SUMMARY_FIELD_KEY]))
    return extreme_datapoint[TIMESPENT_FIELD_KEY]

def remove_outliers(data, minimum_timespent_seconds, maximum_timespent_seconds):

    print("Filtering out datapoints with time spent lower than %d seconds and higher than %d seconds" % (minimum_timespent_seconds, maximum_timespent_seconds))
    filtered_data = [datapoint for datapoint in data
        if datapoint[TIMESPENT_FIELD_KEY] >= minimum_timespent_seconds
            and datapoint[TIMESPENT_FIELD_KEY] <= maximum_timespent_seconds]

    print("%d (%.2f%%) of %d datapoints were selected for testing and training"
        % string_utils.get_part_strings(len(filtered_data), len(data)))

    return filtered_data

def filter_data_by_projects(data, selected_projects):

    if len(selected_projects) == 0:
        return

    selected_data = [datapoint for datapoint in data if projects.is_in(datapoint, selected_projects)]
    print("%d (%.2f%%) of %d datapoints selected" % string_utils.get_part_strings(len(selected_data), len(data)))

    return selected_data

    
def remove_small_projects(data, minimum_project_size):

    issue_counts = projects.get_issue_counts(data)
    selected_projects = {issue_count[0] for issue_count in issue_counts if issue_count[1] >= minimum_project_size}
    print("%d (%.2f%%) of %d projects were selected" % string_utils.get_part_strings(len(selected_projects), len(projects.get(data))))

    return selected_projects

def save_filtered_data(data, dataset_name, labeling):

    filename = get_dataset_filename(dataset_name, labeling, FILTERED_POSTFIX, JSON_FILE_EXTENSION)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=JSON_INDENT)
    
    print("Filtered dataset %s created and saved on %s" % (dataset_name, filename))

def even_distribution(data):

    min_timespent = min(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])[TIMESPENT_FIELD_KEY]
    max_timespent = max(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])[TIMESPENT_FIELD_KEY]
    timespent_range = max_timespent - min_timespent

    bin_count = int(input("Please choose number of bins (timespent range is %.0f hours): " % (timespent_range / SECONDS_IN_HOUR)))
    
    bins, bin_volumes = projects.get_bins_and_volumes(data, bin_count, timespent_range)
    min_bin_volume = min(bin_volumes)
    print("Bin volumes:", *bin_volumes)

    evenly_distributed_data = []
    for i, b in enumerate(bins):
        factor = min_bin_volume / bin_volumes[i]
        for j, d in enumerate(b):
            if round(j * factor) == round((j + 1) * factor):
                continue
            evenly_distributed_data.append(d)

    print("%d (%.2f%%) of %d records were selected and an even distribution was created"
        % string_utils.get_part_strings(len(evenly_distributed_data), len(data)))

    return evenly_distributed_data

def escape_short_texts(data, minimum_text_length):

    filtered_data = [datapoint for datapoint in data if len(datapoint.get(SUMMARY_FIELD_KEY, "").split()) + len(datapoint.get(DESCRIPTION_FIELD_KEY, "").split()) >= minimum_text_length]
    print("%d (%.2f%%) of %d records were selected"
        % string_utils.get_part_strings(len(filtered_data), len(data)))
    return filtered_data


def filter(dataset):

    labeled_data = load_dataset(dataset, LABELED_FILENAME)
    if labeled_data is None:
        print("No labeled data was loaded, filtering cancelled")
        return
    unlabeled_data = load_dataset(dataset, UNLABELED_FILENAME)

    labeled_data = remove_unlabeled_datapoints(labeled_data)

    if input("Would you like to remove tasks with short textual descriptions? (y/n) ") == "y":
        minimum_text_length = int(input("Please enter the minimum text length (words): "))
        labeled_data = escape_short_texts(labeled_data, minimum_text_length)
        unlabeled_data = escape_short_texts(unlabeled_data, minimum_text_length)

    print_extreme(labeled_data, Extreme.MINIMUM)
    maximum = print_extreme(labeled_data, Extreme.MAXIMUM)

    if input("Would you like to remove extreme outliers? (y/n) ") == "y":
        lower_bound_minutes = int(input("Please enter the lower bound (integer) in minutes (input 0 to set no bound): "))
        upper_bound_minutes = int(input("Please enter the upper bound (integer) in minutes (input %s to set no bound): " % ((maximum + SECONDS_IN_MINUTE) // SECONDS_IN_MINUTE)))
        labeled_data = remove_outliers(labeled_data, lower_bound_minutes * SECONDS_IN_MINUTE, upper_bound_minutes * SECONDS_IN_MINUTE)

    if input("Would you like to increase homogeneity by removing small projects? (y/n) ") == "y":
        minimum_project_size = int(input("Please enter the minimum number of labeled filtered datapoints in a project: "))
        selected_projects = remove_small_projects(labeled_data, minimum_project_size)
        labeled_data = filter_data_by_projects(labeled_data, selected_projects)
        unlabeled_data = filter_data_by_projects(unlabeled_data, selected_projects)

    if input("Would you like to make even distribution by removing skewed data? (y/n) ") == "y":
        labeled_data = even_distribution(labeled_data)

    save_filtered_data(labeled_data, dataset, LABELED_FILENAME)
    save_filtered_data(unlabeled_data, dataset, UNLABELED_FILENAME)  

sys_argv_count = len(sys.argv)

if (sys_argv_count < 2):
    print("Please choose a dataset to filter")
    sys.exit()

if (sys_argv_count > 2):
    print("Please choose one argument containing the name of the dataset you want to filter")
    sys.exit()

filter(sys.argv[1])