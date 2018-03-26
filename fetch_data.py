import re
import os
import requests
import sys
import system_constants
import shutil
import xml.etree.ElementTree as et

MAX_RECORDS_PER_REQUEST = 50

def create_directories(respository_name):

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
    print("Request was not successful, status code %d was returned" % response.status_code)

def print_issue_counts(url, auth):

    print("Trying to get the number of issues")

    total_issues = get_number_of_issues(url, auth)
    print("This repository contains %d issues in total." % total_issues)

    total_issues_with_time_spent = get_number_of_issues(url, auth, "timespent > 0")
    labeling_coverage = total_issues_with_time_spent / total_issues * 100
    print("Time spent is reported for %d issues and labeling coverage equals %.2f%%." % (total_issues_with_time_spent, labeling_coverage))
    
    resolved_issues_with_time_spent = get_number_of_issues(url, auth, "timespent > 0 and resolution = 1") 
    resolution_percentage = resolved_issues_with_time_spent / total_issues_with_time_spent * 100
    print("%d (%.2f%%) of the issues with reported time spent are resolved." % (resolved_issues_with_time_spent, resolution_percentage))

    # timespent <= 0 or timespent is EMPTY or resolution != 1

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

    issue_fields = [issue.get("fields") for issue in issues]
    return (issue_fields, total)

def save_slice(filename, data_slice):

    if os.path.isfile(filename):
        tree = et.parse(filename)
        xmlRoot = tree.getroot()
    else:
        xmlRoot = et.Element("items")
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
        save_slice(target_file, data_slice)   

        records_processed = min(startAt + MAX_RECORDS_PER_REQUEST, total_issues)
        print("%d (%.2f%%) of %d issues fetched and saved on %s" % (records_processed, records_processed / total_issues * 100, total_issues, target_file))     

        slice_num = slice_num + 1        

def fetch_data(respository_name, url):

    target_folder = create_directories(respository_name)

    endpoint_url = system_constants.HTTPS_PREFIX + url + system_constants.JIRA_REST + system_constants.JIRA_SEARCH
    auth = get_auth()
    print_issue_counts(endpoint_url, auth)

    labeled_xml_filename = "%s/%s_%s_%s%s" % (target_folder, respository_name, system_constants.LABELED_FILENAME, system_constants.RAW_POSTFIX, system_constants.XML_FILE_EXTENSION)
    fetch_and_save_issues(labeled_xml_filename, endpoint_url, auth, "timespent > 0")

    unlabeled_xml_filename = "%s/%s_%s_%s%s" % (target_folder, respository_name, system_constants.UNLABELED_FILENAME, system_constants.RAW_POSTFIX, system_constants.XML_FILE_EXTENSION)
    fetch_and_save_issues(unlabeled_xml_filename, endpoint_url, auth, "timespent <= 0 or timespent is EMPTY")

sys_argv_count = len(sys.argv)

if sys_argv_count == 3:
    fetch_data(sys.argv[1], sys.argv[2])
    sys.exit()

print("Please pass 2 arguments to this script:\n1. JIRA repository name that will be used as folder name for the retrieved data\n2. JIRA repository URL")
print("Example request: python fetch_data.py myrepo myrepository.atlassian.net")
