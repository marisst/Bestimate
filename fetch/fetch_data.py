import csv
import os
import re
import requests
import sys
import time

from fetch.count_issues import count as get_issue_count
from utilities.load_data import create_dataset_folder
from utilities.constants import *

MAX_RECORDS_PER_REQUEST = 50

def print_issue_counts(repository_search_url, auth):

    print("Fetching the number of total issues")

    total_issues = get_issue_count(repository_search_url, auth)
    total_labeled_issues = get_issue_count(repository_search_url, auth, LABELED_DATA_JQL)
    labeling_coverage = total_labeled_issues / total_issues * 100 if total_issues > 0 else 0
    issue_statement = "This repository contains %d issues in total of which %d (%.2f%%) are labeled."
    print(issue_statement % (total_issues, total_labeled_issues, labeling_coverage))

def fetch_slice(repository_search_url, auth, jql, startAt, maxResults):
    params = {
        "startAt" : startAt,
        "maxResults" : maxResults,
        "fields" : ",".join(FIELD_KEYS),
        "expand" : "",
        "jql" : jql
    }

    requestSucc = False
    timesTried = 0
    while not requestSucc and timesTried < 7:
        try:
            response = requests.get(repository_search_url, params=params, auth=auth)
        except requests.exceptions.RequestException as e:
            print("An exception occured while trying to fetch a slice.")
            print(e)
            timesTried = timesTried + 1
            delay = timesTried * 10 + 2 ** timesTried
            print("Trying again in %d seconds" % delay)
            time.sleep(delay)
            continue
        requestSucc = True


    if response.status_code != 200:
        print("%s returned unexpected status code %d when trying to fetch slice with the following JQL query: %s" % (repository_search_url, response.status_code, jql))

        error_messages = response.json().get("errorMessages")
        if error_messages is not None and len(error_messages) > 0:
            print('\n'.join(error_messages))

    json_response = response.json()
    issues = json_response.get("issues")
    total = json_response.get("total")

    if issues is None:
        return (None, 0)

    issue_fields = [issue.get("fields") for issue in issues]
    return (issue_fields, total)

def save_slice(filename, data_slice):

    data = []

    for datapoint in data_slice:
        element = {}
        for key, field_value in datapoint.items():
            
            if field_value == None:
                continue

            if key == 'project':
                element[key] = field_value.get('key')
            else:
                element[key] = str(field_value)

        data.append(element)

    with open(filename, 'a', newline='') as file:
        csv_writer = csv.writer(file)
        for datapoint in data:
            row = [str(datapoint.get(field, "").encode("utf-8"))[2:-1:] for field in FIELD_KEYS]
            csv_writer.writerow(row)


def fetch_and_save_issues(target_file, repository_search_url, auth, jql=""):

    slice_num = 0
    total_issues = 0

    while slice_num * MAX_RECORDS_PER_REQUEST <= total_issues:

        startAt = slice_num * MAX_RECORDS_PER_REQUEST
        data_slice, total_issues = fetch_slice(repository_search_url, auth, jql, startAt, MAX_RECORDS_PER_REQUEST)
        if total_issues > 0:
            save_slice(target_file, data_slice)   

        records_processed = min(startAt + MAX_RECORDS_PER_REQUEST, total_issues)
        if records_processed > 0:
            processed_percentage = records_processed / total_issues * 100
            print("%d (%.2f%%) of %d issues fetched and saved on %s" % (records_processed, processed_percentage, total_issues, target_file))     

        slice_num = slice_num + 1

    return total_issues

def fetch_data(dataset_name, repository_base_url, auth = None):

    if dataset_name.isdigit():
        print("Digit names are reserved for merged datasets, please choose a different name")
        sys.exit()

    dataset_folder = create_dataset_folder(dataset_name)

    repository_search_url = get_repository_search_url(repository_base_url)
    
    print_issue_counts(repository_search_url, auth)

    labeled_data_filename = get_dataset_filename(dataset_name, LABELED_FILENAME, RAW_POSTFIX, CSV_FILE_EXTENSION)
    labeled_issue_count = fetch_and_save_issues(labeled_data_filename, repository_search_url, auth, LABELED_DATA_JQL)

    unlabeled_data_filename = get_dataset_filename(dataset_name, UNLABELED_FILENAME, RAW_POSTFIX, CSV_FILE_EXTENSION)
    unlabeled_issue_count = fetch_and_save_issues(unlabeled_data_filename, repository_search_url, auth, UNLABELED_DATA_JQL)

    if labeled_issue_count + unlabeled_issue_count > 0:
        print("%d labeled and %d unlabeled issues from %s were fetched and saved in %s" % (labeled_issue_count, unlabeled_issue_count, repository_base_url, dataset_folder))
