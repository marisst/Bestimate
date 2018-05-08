import json
import os
import re

from preprocess import projects
from utilities import input_parser, load_data
from utilities.constants import *

def load_and_parse_data(datasets, labeling):

    data = []
    for dataset in datasets:
        filename = get_data_filename(dataset, labeling, CLEANED_POSTFIX, JSON_FILE_EXTENSION)
        dataset_data = load_data.load_json(filename)

        if dataset_data is None:
            print("%s does not contain labeled datapoints with cleaned text, please run python clean_text.py %s first" % (dataset, dataset))
            continue

        for dataset_datapoint in dataset_data:
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
        if projects.is_in(datapoint, selected_projects):
            filtered_data.append(datapoint)

    percentage = len(filtered_data) / len(data) * 100
    print("%d (%.2f%%) of %d selected" % (len(filtered_data), percentage, len(data)))

    return filtered_data

def exclude_projects(data):

    all_projects = projects.get(data)
    if input("Would you like to exclude any particular projects? (y/n) ") != "y":
        return all_projects

    excluded_projects = input_parser.select_projects(data)
    if len(excluded_projects) == 0:
        print("No projects were excluded")
        return all_projects

    selected_projects = all_projects - excluded_projects
    return selected_projects

def select_projects(data):

    if data is None:
        return

    print("You will be able to select the minimum number of issues in a project later")
    if input("Do you want to train and test only on selected projects? (y/n) ") != "y":
        return exclude_projects(data)
        
    selected_projects = input_parser.select_projects(data)
    if len(selected_projects) == 0:
        print("No projects were selected")
        return

    return selected_projects

def save_merged_data(data, dataset_name, labeling):

    filename = get_dataset_filename(dataset_name, labeling, MERGED_POSTFIX, JSON_FILE_EXTENSION)

    with open(filename, 'w') as file:
        json.dump(data, file, indent=JSON_INDENT)
    
    print("Merged dataset %s created and saved on %s" % (dataset_name, filename))

def merge_data(datasets_from_input):

    datasets = input_parser.select_datasets(datasets_from_input)
    
    if len(datasets) > 0:
        print("Training and testing model on the following dataset%s:" % ("s" if len(datasets) > 1 else ""), ", ".join(datasets))
    else:
        print("No datasets selected")
        return

    labeled_data = load_and_parse_data(datasets, LABELED_FILENAME)
    unlabeled_data = load_and_parse_data(datasets, UNLABELED_FILENAME)

    selected_projects = select_projects(labeled_data)
    if selected_projects is None or len(selected_projects) == 0:
        print("No projects selected, merge is cancelled")
        return
    print("Merging data from the following projects:", *selected_projects)

    labeled_data = filter_by_projects(labeled_data, selected_projects)
    unlabeled_data = filter_by_projects(unlabeled_data, selected_projects)

    if labeled_data is None:
        print("No labeled data was selected, merge is cancelled")
        return

    load_data.create_folder_if_needed(DATASET_FOLDER)
    dataset_name = load_data.get_next_dataset_name(DATASET_FOLDER)
    load_data.create_dataset_folder(dataset_name, DATASET_FOLDER)

    save_merged_data(labeled_data, dataset_name, LABELED_FILENAME)
    save_merged_data(unlabeled_data, dataset_name, UNLABELED_FILENAME)