import json
import os
import re

from utilities.constants import get_repository_filename, get_dataset_filename
from utilities.data_utils import get_projects, is_in_projects
from utilities.file_utils import create_subfolder, get_next_subfolder_name, load_json, save_json
from utilities.constants import ALPHA_FIELD, CLEANED_POSTFIX, DATASET_FOLDER, DESCRIPTION_FIELD_KEY, JSON_FILE_EXTENSION, LABELED_FILENAME
from utilities.constants import MERGED_POSTFIX, PROJECT_FIELD_KEY, SUMMARY_FIELD_KEY, TIMESPENT_FIELD_KEY, UNLABELED_FILENAME
from utilities.input_parser import select_repositories, select_projects


def load_and_parse_data(datasets, labeling):

    data = []
    for dataset in datasets:
        filename = get_repository_filename(dataset, labeling, CLEANED_POSTFIX, JSON_FILE_EXTENSION)
        dataset_data = load_json(filename)

        if dataset_data is None:
            print("%s does not contain labeled datapoints with cleaned text, please run python clean_text.py %s first" % (dataset, dataset))
            continue

        for dataset_datapoint in dataset_data:

            if dataset_datapoint.get(SUMMARY_FIELD_KEY) is None:
                continue

            training_datapoint = {
                PROJECT_FIELD_KEY : "%s-%s" % (dataset, dataset_datapoint[PROJECT_FIELD_KEY]),
                SUMMARY_FIELD_KEY : dataset_datapoint[SUMMARY_FIELD_KEY]
            }
            if DESCRIPTION_FIELD_KEY in dataset_datapoint:
                training_datapoint[DESCRIPTION_FIELD_KEY] = dataset_datapoint[DESCRIPTION_FIELD_KEY]
            if TIMESPENT_FIELD_KEY in dataset_datapoint:
                training_datapoint[TIMESPENT_FIELD_KEY] = int(dataset_datapoint[TIMESPENT_FIELD_KEY])
            if ALPHA_FIELD in dataset_datapoint:
                training_datapoint[ALPHA_FIELD] = dataset_datapoint[ALPHA_FIELD]

            data.append(training_datapoint)

    if len(data) == 0:
        print("No data was selected")
        return

    return data


def filter_by_projects(data, selected_projects):

    filtered_data = []
    for datapoint in data:
        if is_in_projects(datapoint, selected_projects):
            filtered_data.append(datapoint)

    percentage = len(filtered_data) / len(data) * 100
    print("%d (%.2f%%) of %d selected" % (len(filtered_data), percentage, len(data)))

    return filtered_data


def exclude_projects(data):

    all_projects = get_projects(data)
    if input("Would you like to exclude any particular projects? (y/n) ") != "y":
        return all_projects

    excluded_projects = select_projects(data)
    if len(excluded_projects) == 0:
        print("No projects were excluded")
        return all_projects

    selected_projects = all_projects - excluded_projects
    return selected_projects


def select_or_exclude_projects(data):
    """Let user select of exclude particular projects from the pool"""

    if data is None:
        return

    print("You will be able to select the minimum number of issues in a project later")
    if input("Do you want to train and test only on selected projects? (y/n) ") != "y":
        return exclude_projects(data)
        
    selected_projects = select_projects(data)
    if len(selected_projects) == 0:
        print("No projects were selected")
        return

    return selected_projects


def save_merged_data(data, dataset_name, labeling):

    filename = get_dataset_filename(dataset_name, labeling, MERGED_POSTFIX, JSON_FILE_EXTENSION)
    save_json(filename, data)  
    print("Merged dataset %s created and saved on %s" % (dataset_name, filename))


def merge_data(repository_identifiers, enable_manual_project_selection = False):
    """Merge data from several repositories, select or exclude projects
    and save as a new training dataset

    Arguments:

    repository_identifiers -- repository identifiers that are to get merged

    enable_manual_project_selection -- allow user to select particular projects,
    or exclude particular project throught command line interface (default False)
    """

    repositories = select_repositories(repository_identifiers)
    
    if len(repositories) > 0:
        print("Merging the following repositories:", ", ".join(repositories))
    else:
        print("No repositories selected")
        return

    labeled_data = load_and_parse_data(repositories, LABELED_FILENAME)
    unlabeled_data = load_and_parse_data(repositories, UNLABELED_FILENAME)

    if enable_manual_project_selection == True:
        selected_projects = select_or_exclude_projects(labeled_data + unlabeled_data)
    else:
        selected_projects = get_projects(labeled_data + unlabeled_data)

    if selected_projects is None or len(selected_projects) == 0:
        print("No projects selected, merge is cancelled")
        return
    print("Merging data from the following projects:", *selected_projects)

    labeled_data = filter_by_projects(labeled_data, selected_projects)
    unlabeled_data = filter_by_projects(unlabeled_data, selected_projects)

    if labeled_data is None:
        print("No labeled data was selected, merge is cancelled")
        return

    dataset_name = get_next_subfolder_name(DATASET_FOLDER)
    create_subfolder(DATASET_FOLDER, dataset_name)
    save_merged_data(labeled_data, dataset_name, LABELED_FILENAME)
    save_merged_data(unlabeled_data, dataset_name, UNLABELED_FILENAME)

    return dataset_name


if __name__ == "__main__":
    
    repository_identifiers = input("List one or more repository identifiers which you want to merge or leave blank and press ENTER to merge all cleaned data: ")
    merge_data(repository_identifiers, True)