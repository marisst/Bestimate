from enum import Enum
import json
import sys

from preprocessing import projects
from utilities import load_data, string_utils
from utilities.constants import *

class Extreme(Enum):
    MINIMUM = min
    MAXIMUM = max

def load_dataset(dataset):
    
    filename = get_merged_dataset_filename(dataset)
    return load_data.load_json(filename)

def print_extreme(data, extreme):

    extreme_datapoint = extreme.value(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])
    print("The %s timespent is %.2f hours for the following issue: %s"
        % (
            extreme.name,
            extreme_datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR,
            extreme_datapoint[SUMMARY_FIELD_KEY]))
    return extreme_datapoint[TIMESPENT_FIELD_KEY]

def remove_outliers(data, minimum_timespent_seconds, maximum_timespent_seconds):

    print("Filtering out datapoints with time spent lower than %d minutes and higher than %d minutes" % (minimum_timespent_seconds, maximum_timespent_seconds))
    filtered_data = [datapoint for datapoint in data
        if datapoint[TIMESPENT_FIELD_KEY] >= minimum_timespent_seconds
            and datapoint[TIMESPENT_FIELD_KEY] <= maximum_timespent_seconds]

    print("%d (%.2f%%) of %d datapoints were selected for testing and training"
        % string_utils.get_part_strings(len(filtered_data), len(data)))
    
def remove_small_projects(data, minimum_project_size):

    issue_counts = projects.get_issue_counts(data)
    selected_projects = {issue_count[0] for issue_count in issue_counts if issue_count[1] >= minimum_project_size}
    print("%d (%.2f%%) of %d projects were selected" % string_utils.get_part_strings(len(selected_projects), len(projects.get(data))))

    if len(selected_projects) == 0:
        return

    selected_data = [datapoint for datapoint in data if projects.is_in(datapoint, selected_projects)]
    print("%d (%.2f%%) of %d datapoints selected" % string_utils.get_part_strings(len(selected_data), len(data)))

    return data

def save_filtered_data(data, dataset_name):

    load_data.create_folder_if_needed(FILTERED_DATA_FOLDER)
    filename = get_filtered_dataset_filename(dataset_name)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=JSON_INDENT)
    
    print("Filtered dataset %s created and saved on %s" % (dataset_name, filename))

def filter(dataset):

    data = load_dataset(dataset)
    if data is None:
        return

    print_extreme(data, Extreme.MINIMUM)
    maximum = print_extreme(data, Extreme.MAXIMUM)

    if input("Would you like to remove extreme outliers? (y/n) ") == "y":
        lower_bound_minutes = int(input("Please enter the lower bound (integer) in minutes (input 0 to set no bound): "))
        upper_bound_minutes = int(input("Please enter the upper bound (integer) in minutes (input %s to set no bound): " % ((maximum + SECONDS_IN_MINUTE) // SECONDS_IN_MINUTE)))
        data = remove_outliers(data, lower_bound_minutes * SECONDS_IN_MINUTE, upper_bound_minutes * SECONDS_IN_MINUTE)

    if input("Would you like to increase homogeneity by removing small projects? (y/n) ") == "y":
        minimum_project_size = int(input("Please enter the minimum number of labeled filtered datapoints in a project: "))
        data = remove_small_projects(data, minimum_project_size)

    save_filtered_data(data, dataset)
        

sys_argv_count = len(sys.argv)

if (sys_argv_count < 2):
    print("Please choose a dataset to filter")
    sys.exit()

if (sys_argv_count > 2):
    print("Please choose one argument containing the name of the dataset you want to filter")
    sys.exit()

filter(sys.argv[1])