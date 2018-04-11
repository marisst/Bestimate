import os
import re

from preprocessing import projects
from utilities.constants import *

def select_datasets(datasets):

    # selecting all datasets
    if len(datasets) == 0:
        return [entry for entry in os.listdir(DATA_FOLDER) if os.path.isdir("%s/%s" % (DATA_FOLDER, entry))]

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