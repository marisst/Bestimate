import os
import re

from preprocess import projects
from utilities.constants import *

def select_datasets(datasets):

    # selecting all datasets
    if len(datasets) == 0:
        data_folder_subfolders = [entry for entry in os.listdir(DATA_FOLDER) if os.path.isdir("%s/%s" % (DATA_FOLDER, entry))]
        non_dataset_folders = [FILTERED_DATA_FOLDER, STATISTICS_FOLDER]
        non_dataset_folders = [x[len(DATA_FOLDER)+1:] for x in non_dataset_folders]
        return [x for x in data_folder_subfolders if x not in non_dataset_folders and not x.isdigit()]

    for dataset in datasets:
        if not os.path.isdir("%s/%s" % (DATA_FOLDER, dataset)):
            print("Dataset %s does not exist" % dataset)
            datasets.remove(dataset)

    return datasets

def select_projects(data):

    print("Please select one or more of the following projects:")
    project_issue_counts = projects.get_issue_counts(data)

    for c in project_issue_counts:
        print("%s - %d issues" % c)

    selected_projects = input("Selected datasets: ")
    selected_projects = selected_projects.replace(",", " ")
    selected_projects = re.sub(r"[^ A-Za-z1-9\-]", "", selected_projects)
    selected_projects = set(selected_projects.split())
    
    return selected_projects & projects.get(data)