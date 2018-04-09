from constants import *

def get(data):

    if data is None:
        return

    return {datapoint[PROJECT_FIELD_KEY] for datapoint in data}

def get_issue_counts(data):

    project_list = get(data)
    if project_list is None:
        return

    project_issue_counts = []
    for project in project_list:
        issue_count = sum(1 for datapoint in data if datapoint[PROJECT_FIELD_KEY] == project)
        project_issue_counts.append((project, issue_count))

    return sorted(project_issue_counts, key = lambda a: a[1])

def is_in(datapoint, selected_projects):

    return len({datapoint[PROJECT_FIELD_KEY]}.intersection(selected_projects)) > 0
