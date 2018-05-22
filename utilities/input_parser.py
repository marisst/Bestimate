import os
import re

from utilities.data_utils import get_issue_counts, get_projects
from utilities.string_utils import get_part_strings
from utilities.constants import *

def select_repositories(repositories_from_input):

    if len(repositories_from_input) == 0:
        data_folder_subfolders = [entry for entry in os.listdir(DATA_FOLDER) if os.path.isdir("%s/%s" % (DATA_FOLDER, entry))]
        non_dataset_folders = [STATISTICS_FOLDER]
        non_dataset_folders = [x[len(DATA_FOLDER)+1:] for x in non_dataset_folders]
        return [x for x in data_folder_subfolders if x not in non_dataset_folders and not x.isdigit()]

    repositories_from_input = repositories_from_input.split(" ")
    for dataset in repositories_from_input:
        if not os.path.isdir("%s/%s" % (DATA_FOLDER, dataset)):
            print("Dataset %s does not exist" % dataset)
            repositories_from_input.remove(dataset)

    return repositories_from_input

def select_projects(data):

    print("Please select one or more of the following projects:")
    project_issue_counts = get_issue_counts([datapoint for datapoint in data if datapoint.get(TIMESPENT_FIELD_KEY, None) is not None])

    for c in project_issue_counts:
        total_issue_count = sum([1 for datapoint in data if datapoint.get(PROJECT_FIELD_KEY) == c[0]])
        part_strings = get_part_strings(c[1], total_issue_count)
        print("%s - %d (%.2f%%) of %d issues are labeled" % (c[0], part_strings[0], part_strings[1], part_strings[2]))

    selected_projects = input("Selected datasets: ")
    selected_projects = selected_projects.replace(",", " ")
    selected_projects = re.sub(r"[^ A-Za-z1-9\-]", "", selected_projects)
    selected_projects = set(selected_projects.split())
    
    return selected_projects & get_projects(data)