from enum import Enum
import json
import sys

from utilities.constants import get_dataset_filename
from utilities.constants import DESCRIPTION_FIELD_KEY, FILTERED_POSTFIX, JSON_FILE_EXTENSION, LABELED_FILENAME, MERGED_POSTFIX
from utilities.constants import SECONDS_IN_HOUR, SECONDS_IN_MINUTE, SUMMARY_FIELD_KEY, TIMESPENT_FIELD_KEY, UNLABELED_FILENAME
from utilities.data_utils import get_issue_counts, get_projects, is_in_projects, get_bins_and_volumes
from utilities.file_utils import load_json, save_json
from utilities.string_utils import merge_sentences, get_part_strings, word_count
from data_preprocessing.filter_config import FilterConfig


def load_dataset(dataset, labeling):
    
    filename = get_dataset_filename(dataset, labeling, MERGED_POSTFIX, JSON_FILE_EXTENSION)
    return load_json(filename)


def remove_unlabeled_datapoints(data):

    labeled_data = [datapoint for datapoint in data if TIMESPENT_FIELD_KEY in datapoint]
    if(len(labeled_data) != len(data)):
        print("%d (%d%%) of %d datapoints were removed because they were unlabeled" % get_part_strings(len(data)-len(labeled_data), len(data)))
    return labeled_data


def get_unlabeled_datapoints(data):

    return [datapoint for datapoint in data if TIMESPENT_FIELD_KEY not in datapoint]


def remove_outliers(data, minimum_timespent_seconds, maximum_timespent_seconds):

    print("Filtering out datapoints with time spent lower than %d seconds and higher than %d seconds" % (minimum_timespent_seconds, maximum_timespent_seconds))
    filtered_data = [datapoint for datapoint in data
        if datapoint[TIMESPENT_FIELD_KEY] >= minimum_timespent_seconds
            and datapoint[TIMESPENT_FIELD_KEY] <= maximum_timespent_seconds]

    print("%d (%.2f%%) of %d datapoints were selected for testing and training" % get_part_strings(len(filtered_data), len(data)))

    return filtered_data


def filter_data_by_projects(data, selected_projects):

    if len(selected_projects) == 0:
        return

    selected_data = [datapoint for datapoint in data if is_in_projects(datapoint, selected_projects)]
    print("%d (%.2f%%) of %d datapoints selected" % get_part_strings(len(selected_data), len(data)))

    return selected_data

    
def remove_small_projects(data, minimum_project_size):

    issue_counts = get_issue_counts(data)
    selected_projects = {issue_count[0] for issue_count in issue_counts if issue_count[1] >= minimum_project_size}
    print("%d (%.2f%%) of %d projects were selected" % get_part_strings(len(selected_projects), len(get_projects(data))))

    return selected_projects


def save_filtered_data(data, dataset_name, labeling):

    filename = get_dataset_filename(dataset_name, labeling, FILTERED_POSTFIX, JSON_FILE_EXTENSION)
    save_json(filename, data)    
    print("Filtered dataset %s created and saved on %s" % (dataset_name, filename))


def even_distribution(data, bin_count):
    """Create even distribution by removing data from bins with higher datapoint count than the smallest bin"""

    if data is None:
        return []

    min_timespent = min(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])[TIMESPENT_FIELD_KEY]
    max_timespent = max(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])[TIMESPENT_FIELD_KEY]
    timespent_range = max_timespent - min_timespent
    
    bins, bin_volumes = get_bins_and_volumes(data, bin_count, timespent_range)
    min_bin_volume = min(bin_volumes)
    print("Bin volumes:", *bin_volumes)

    evenly_distributed_data = []
    for i, b in enumerate(bins):
        factor = min_bin_volume / bin_volumes[i]
        for j, d in enumerate(b):
            if round(j * factor) == round((j + 1) * factor):
                continue
            evenly_distributed_data.append(d)

    print("%d (%.2f%%) of %d records were selected and an even distribution was created" % get_part_strings(len(evenly_distributed_data), len(data)))

    return evenly_distributed_data


def escape_short_texts(data, minimum_words):
    """Remove task with description length shorter than minimum_words"""

    filtered_data = [datapoint for datapoint in data if word_count(datapoint.get(SUMMARY_FIELD_KEY, "")) + word_count(datapoint.get(DESCRIPTION_FIELD_KEY, "")) >= minimum_words]
    print("%d (%.2f%%) of %d records were selected" % get_part_strings(len(filtered_data), len(data)))
    return filtered_data


def filter_data(dataset, filter_config, notes_filename = None):
    """Filter data of a merged dataset according to a filter configuration and save in JSON format"""

    print("Loading data...")
    labeled_data = load_dataset(dataset, LABELED_FILENAME)
    if labeled_data is None:
        print("No labeled data was loaded, filtering cancelled")
        return
    unlabeled_data = load_dataset(dataset, UNLABELED_FILENAME)

    unlabeled_labeled_data = get_unlabeled_datapoints(labeled_data)
    if (len(unlabeled_labeled_data) > 0):
        print("Processing unlabeled datapoints, which are marked as labeled...")
        unlabeled_data = unlabeled_data + unlabeled_labeled_data
        print("%d unlabeled datapoints marked as labeled moved to unlabeled dataset" % len(unlabeled_labeled_data))
    labeled_data = remove_unlabeled_datapoints(labeled_data)

    if notes_filename is not None:
        with open(notes_filename, "a") as notes_file:
            print("%d labeled and %d unlabeled issues before filtering" % (len(labeled_data), len(unlabeled_data)), file=notes_file)

    if filter_config.min_word_count > 0:
        print("Removing datapoints with short text descriptions...")
        labeled_data = escape_short_texts(labeled_data, filter_config.min_word_count)
        unlabeled_data = escape_short_texts(unlabeled_data, filter_config.min_word_count)

    if filter_config.min_timespent_minutes > 0 or filter_config.max_timespent_minutes < sys.maxsize:
        print("Removing outliers...")
        labeled_data = remove_outliers(labeled_data, filter_config.min_timespent_minutes * SECONDS_IN_MINUTE, filter_config.max_timespent_minutes * SECONDS_IN_MINUTE)

    if filter_config.min_project_size > 0:
        print("Removing small projects...")
        selected_projects = remove_small_projects(labeled_data, filter_config.min_project_size)
        labeled_data = filter_data_by_projects(labeled_data, selected_projects)
        unlabeled_data = filter_data_by_projects(unlabeled_data, selected_projects)

    if filter_config.even_distribution_bin_count > 0:
        print("Flattening distribution...")
        labeled_data = even_distribution(labeled_data, filter_config.even_distribution_bin_count)

    print("Saving filtered data...")
    save_filtered_data(labeled_data, dataset, LABELED_FILENAME)
    save_filtered_data(unlabeled_data, dataset, UNLABELED_FILENAME)

    if notes_filename is not None:
        with open(notes_filename, "a") as notes_file:
            print("%d labeled and %d unlabeled issues after filtering" % (len(labeled_data), len(unlabeled_data)), file=notes_file)

def print_extreme(dataset, extreme):
    "Print the minimum or maximum number of hours of time spent for a task in a merged dataset"

    print("Getting label range...")
    data = load_dataset(dataset, LABELED_FILENAME)
    data = [datapoint for datapoint in data if datapoint.get(TIMESPENT_FIELD_KEY) is not None]
    extreme_datapoint = extreme.value(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])
    print("The %s timespent is %.2f hours for the following issue: %s"
        % (
            extreme.name,
            extreme_datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR,
            merge_sentences(extreme_datapoint[SUMMARY_FIELD_KEY])))


class Extreme(Enum):
    MINIMUM = min
    MAXIMUM = max


if __name__ == "__main__":

    training_dataset_name = input("Please enter the name of the training dataset you wish to filter: ")
    filter_config = FilterConfig()

    if input("Would you like to remove tasks with short textual descriptions? (y/n) ") == "y":
        filter_config.min_word_count = int(input("Please enter the minimum text length (words): "))

    print_extreme(training_dataset_name, Extreme.MINIMUM)
    print_extreme(training_dataset_name, Extreme.MAXIMUM)
    if input("Would you like to remove extreme outliers? (y/n) ") == "y":
        filter_config.min_timespent_minutes = int(input("Please enter the lower bound (integer) in minutes: "))
        filter_config.max_timespent_minutes = int(input("Please enter the upper bound (integer) in minutes: "))

    if input("Would you like to increase homogeneity by removing small projects? (y/n) ") == "y":
        filter_config.min_project_size = int(input("Please enter the minimum number of labeled filtered datapoints in a project: "))
        
    if input("Would you like to make even distribution by removing skewed data? (y/n) ") == "y":
        filter_config.even_distribution_bin_count = int(input("Please choose number of bins: "))

    filter_data(training_dataset_name, filter_config)