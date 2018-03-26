import re
import os
import requests
import sys
import system_constants
import shutil
import xml.etree.ElementTree as et

MAX_RECORDS_PER_REQUEST = 50
LABELED_DATA_JQL = "timespent > 0 and resolution = 1"
UNLABELED_DATA_JQL = "timespent <= 0 or timespent is EMPTY or and resolution != 1"

def create_folder(respository_name):

    if not os.path.exists(system_constants.DATA_FOLDER):
        os.makedirs(system_constants.DATA_FOLDER)
    
    repository_data_folder = "%s/%s" % (system_constants.DATA_FOLDER, respository_name)
    if os.path.exists(repository_data_folder):
        if input("%s already exists, do you want to remove it's contents? (y/n) " % repository_data_folder) != "y":
            sys.exit()
        shutil.rmtree(repository_data_folder)
    os.makedirs(repository_data_folder)

    return repository_data_folder

def get_auth():

    authorize = input("Do you want to sign in? (y/n) ") == "y"

    if authorize:
        username = input("Username: ")
        api_token = input("API token: ")
    auth = (username, api_token) if authorize else None

def get_number_of_issues(url, auth, jql=""):
    
    params = {
        "maxResults" : "0",
        "jql" : jql
    }
    response = requests.get(url, params=params, auth=auth)

    if response.status_code == 200:
        return response.json().get("total")
    return 0

def print_issue_counts(url, auth):

    print("Fetching the number of total issues")

    total_issues = get_number_of_issues(url, auth)
    total_labeled_issues = get_number_of_issues(url, auth, LABELED_DATA_JQL)
    labeling_coverage = total_labeled_issues / total_issues * 100 if total_issues > 0 else 0
    print("This repository contains %d issues in total of which %d (%.2f%%) of the issues are labeled." % (total_issues, total_labeled_issues, labeling_coverage))

def fetch_slice(url, auth, jql, startAt, maxResults):
    params = {
        "startAt" : startAt,
        "maxResults" : maxResults,
        "fields" : "summary,description,timespent",
        "expand" : "",
        "jql" : jql
    }

    json_response = requests.get(url, params=params, auth=auth).json()

    issues = json_response.get("issues")
    total = json_response.get("total")

    if issues is None:
        return (None, 0)

    issue_fields = [issue.get("fields") for issue in issues]
    return (issue_fields, total)

def save_slice(filename, data_slice):

    if os.path.isfile(filename):
        tree = et.parse(filename)
        xmlRoot = tree.getroot()
    else:
        xmlRoot = et.Element(system_constants.XML_ROOT_NAME)
        tree = et.ElementTree(xmlRoot)

    for datapoint in data_slice:
        item = et.Element(system_constants.XML_ITEM_NAME)

        for key, field_value in datapoint.items():
            
            if field_value == None:
                continue
            
            feature = et.SubElement(item, key)

            # convert to string and escape invelid XML characters
            feature.text = re.sub(u'[^\u0020-\uD7FF\u0009\u000A\u000D\uE000-\uFFFD\U00010000-\U0010FFFF]+', '', str(field_value))

        xmlRoot.append(item)

    tree.write(filename)

def fetch_and_save_issues(target_file, url, auth, jql=""):

    slice_num = 0
    total_issues = 0

    while slice_num * MAX_RECORDS_PER_REQUEST <= total_issues:

        startAt = slice_num * MAX_RECORDS_PER_REQUEST
        data_slice, total_issues = fetch_slice(url, auth, jql, startAt, MAX_RECORDS_PER_REQUEST)
        if total_issues > 0:
            save_slice(target_file, data_slice)   

        records_processed = min(startAt + MAX_RECORDS_PER_REQUEST, total_issues)
        if records_processed > 0:
            processed_percentage = records_processed / total_issues * 100
            print("%d (%.2f%%) of %d issues fetched and saved on %s" % (records_processed, processed_percentage, total_issues, target_file))     

        slice_num = slice_num + 1

    return total_issues     

def fetch_data(respository_name, url):

    target_folder = create_folder(respository_name)

    endpoint_url = system_constants.HTTPS_PREFIX + url + system_constants.JIRA_REST + system_constants.JIRA_SEARCH
    auth = get_auth()
    print_issue_counts(endpoint_url, auth)

    labeled_xml_filename = "%s/%s_%s_%s%s" % (target_folder, respository_name, system_constants.LABELED_FILENAME, system_constants.RAW_POSTFIX, system_constants.XML_FILE_EXTENSION)
    saved_labeled_issues = fetch_and_save_issues(labeled_xml_filename, endpoint_url, auth, LABELED_DATA_JQL)

    unlabeled_xml_filename = "%s/%s_%s_%s%s" % (target_folder, respository_name, system_constants.UNLABELED_FILENAME, system_constants.RAW_POSTFIX, system_constants.XML_FILE_EXTENSION)
    saved_unlabeled_issues = fetch_and_save_issues(unlabeled_xml_filename, endpoint_url, auth, UNLABELED_DATA_JQL)

    if saved_labeled_issues + saved_unlabeled_issues > 0:
        print("%d labeled and %d unlabeled issues from %s were fetched and saved in %s" % (saved_labeled_issues, saved_unlabeled_issues, url, target_folder))

sys_argv_count = len(sys.argv)

if sys_argv_count == 3:
    fetch_data(sys.argv[1], sys.argv[2])
    sys.exit()

print("Please pass 2 arguments to this script:\n1. JIRA repository name that will be used as folder name for the retrieved data\n2. JIRA repository URL")
print("Example request: python fetch_data.py myrepo myrepository.atlassian.net")