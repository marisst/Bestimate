from utilities.constants import *

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

    return len({datapoint[PROJECT_FIELD_KEY]} & selected_projects) > 0

def get_bins_and_volumes(data, bin_count, timespent_range):

    bins = []
    bin_range = timespent_range / bin_count
    for i in range(bin_count):

        from_timespent = bin_range * i
        to_timespent = bin_range * (i + 1)
        bins.append([datapoint for datapoint in data if datapoint[TIMESPENT_FIELD_KEY] > from_timespent and datapoint[TIMESPENT_FIELD_KEY] <= to_timespent])
    bin_volumes = [len(b) for b in bins]

    return (bins, bin_volumes)
