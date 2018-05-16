import csv
import os
import re
import requests
import sys
import time

from data_collection.test_repos import get_issue_count, get_jira_base_url
from utilities.constants import CSV_FILE_EXTENSION, DATA_FOLDER, FIELD_KEYS, LABELED_DATA_JQL, LABELED_FILENAME, RAW_POSTFIX, UNLABELED_FILENAME, UNLABELED_DATA_JQL
from utilities.file_utils import create_subfolder, get_repository_search_url, get_data_filename

MAX_RECORDS_PER_REQUEST = 50


def fetch_slice(repository_search_url, auth, jql, start_at, max_results):
    """Fetch a chunk of issues from JIRA repository

    Arguments:

    repository_search_url -- search interface endpoint address of JIRA REST API

    auth -- authentication parameters containing username and API key or password,
    None if authentication is not necessary

    jql -- JIRA query if issues need to be filtered

    start_at -- issue start index

    max_results -- maximum number of issues to be fetched
    """

    params = {
        "startAt" : start_at,
        "maxResults" : max_results,
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

    try:
        json_response = response.json()
        issues = json_response.get("issues")
        total = json_response.get("total")
    except Exception as e:
        print("Could not retrieve issues from the response, exception occured:", e)
        return (None, 0)

    if issues is None:
        return (None, 0)

    issue_fields = [issue.get("fields") for issue in issues]
    return (issue_fields, total)


def save_slice(filename, data_slice):
    """Append a list of JIRA issues to a CSV file

    Arguments:

    filename - the name of the CSV file

    data_slice - a list of JIRA issues
    """

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


def fetch_and_save_issues(target_filename, repository_search_url, auth, jql=""):
    """Fetch issues using JIRA REST API in slices of 50 requests and save in CSV format

    Arguments:

    target_filename -- the name of the JSON file in which the issues are to ba saved

    repository_search_url -- search interface endpoint address of JIRA REST API

    auth -- authentication parameters containing username and API key or password,
    None if authentication is not necessary

    jql -- JIRA query if issues need to be filtered
    """

    slice_num = 0
    total_issues = 0

    while slice_num * MAX_RECORDS_PER_REQUEST <= total_issues:

        startAt = slice_num * MAX_RECORDS_PER_REQUEST
        data_slice, total_issues = fetch_slice(repository_search_url, auth, jql, startAt, MAX_RECORDS_PER_REQUEST)
        if total_issues > 0:
            save_slice(target_filename, data_slice)   

        records_processed = min(startAt + MAX_RECORDS_PER_REQUEST, total_issues)
        if records_processed > 0:
            processed_percentage = records_processed / total_issues * 100
            print("%d (%.2f%%) of %d issues fetched and saved at %s" % (records_processed, processed_percentage, total_issues, target_filename))     

        slice_num = slice_num + 1

    return total_issues


def fetch_data(repository_identifier, repository_url, auth = None):
    """Fetch labeled and unlabeled issues from JIRA repository and save in CSV format

    Arguments:

    repository_identifier -- the name of a subfolder in raw_data folder where the data will be saved

    repository_url -- the URL of the repository from which data is to be fetched e.g. jira.exoplatform.org

    auth -- authentication parameters containing username and API key or password (default None)
    """

    folder = create_subfolder(DATA_FOLDER, repository_identifier)
    repository_base_url = get_jira_base_url(repository_url)
    repository_search_url = get_repository_search_url(repository_base_url)
    print_issue_counts(repository_search_url, auth)

    issue_counts = {}
    for labeling in [(LABELED_FILENAME, LABELED_DATA_JQL), (UNLABELED_FILENAME, UNLABELED_DATA_JQL)]:
        filename = get_data_filename(repository_identifier, labeling[0], RAW_POSTFIX, CSV_FILE_EXTENSION)
        issue_counts[labeling[0]] = fetch_and_save_issues(filename, repository_search_url, auth, labeling[1])

    if issue_counts[LABELED_FILENAME] + issue_counts[UNLABELED_FILENAME] > 0:
        print("%d labeled and %d unlabeled issues from %s were fetched and saved at %s"
            % (issue_counts[LABELED_FILENAME], issue_counts[UNLABELED_FILENAME], repository_base_url, folder))


def print_issue_counts(repository_search_url, auth):
    """Fetch and print the number of labeled and unlabeled issues in a JIRA repository

    Arguments:

    repository_search_url -- search interface endpoint address of JIRA REST API

    auth -- authentication parameters containing username and API key or password,
    None if authentication is not necessary
    """

    print("Fetching the number of total issues")
    total_issues = get_issue_count(repository_search_url, auth)
    total_labeled_issues = get_issue_count(repository_search_url, auth, LABELED_DATA_JQL)
    labeling_coverage = total_labeled_issues / total_issues * 100 if total_issues > 0 else 0
    issue_statement = "This repository contains %d issues in total of which %d (%.2f%%) are labeled."
    print(issue_statement % (total_issues, total_labeled_issues, labeling_coverage))


def get_auth():
    """Ask user to input username and API token or password if they choose to authorize"""

    authorize = input("Do you want to sign in? (y/n) ") == "y"

    if authorize:
        username = input("Username: ")
        api_token = input("API token or password: ")

    return (username, api_token) if authorize is True else None


if __name__ == "__main__":

    repository_url = input("Please enter the URL of the repository (e.g. jira.exoplatform.org): ")
    dataset_identifier = input("Please enter an identifier for the repository (only letters and numbers): ")
    auth = get_auth()
    fetch_data(dataset_identifier, repository_url, auth)