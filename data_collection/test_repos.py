import json
import re
import requests
import sys

from utilities.constants import get_repository_search_url, LABELED_DATA_JQL, POTENTIAL_REPOS_FILENAME, REQUEST_TIMEOUT_SECONDS, TIMESPENT_FIELD_KEY 

def get_issue_count(repository_search_url, auth=None, jql=""):
    """Get the number of JIRA issues in JIRA repository.
    
    Arguments:\n
    repository_search_url -- JIRA repository REST API search endpoint URL, e.g. 'https://jira.exoplatform.org/rest/api/latest/search'\n
    auth -- authentication parameters including username and API key or password can optionally be passed to gain access to more issues\n
    jql -- a JQL query narrowing down the issue pool, such as 'timespent > 0' to return the count of all issues with time spent greater than zero
    """

    params = {
        "maxResults" : "0",
        "jql" : jql
    }

    try:
        response = requests.get(repository_search_url, params=params, auth=auth, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.RequestException:
        print("An exception occurred while trying to get issue count")
        return 0

    if response.status_code != 200:
        print("%s returned unexpected status code %d when trying to get number of issues with the following JQL query: %s" % (repository_search_url, response.status_code, jql))
        
        try:
            error_messages = response.json().get("errorMessages")
        except json.decoder.JSONDecodeError:
            print("Could not decode error message")
            return 0

        if error_messages is not None and len(error_messages) > 0:
            print('\n'.join(error_messages))

        return 0

    try: 
        return response.json().get("total")
    except json.decoder.JSONDecodeError:
        print("Response did not contain issue count")
        return 0

def is_timespent_returned(repository_search_url):
    """Check if 'timespent' field is publicly accessible in a JIRA repository
    
    Arguments:\n
    repository_search_url -- JIRA repository REST API search endpoint URL, e.g. 'https://jira.exoplatform.org/rest/api/latest/search'
    """
    
    params = {
        "maxResults": 1,
        "fields": TIMESPENT_FIELD_KEY,
        "expand": "",
        "jql": LABELED_DATA_JQL
    }

    try:
        response = requests.get(repository_search_url, params=params)
    except requests.exceptions.RequestException:
        return False

    if response.status_code != 200:
        return False

    try:
        json_response = response.json()
    except json.JSONDecodeError:
        return False

    issues = json_response.get("issues", None)
    if issues is None or len(issues) < 1:
        return False

    fields = issues[0].get("fields", None)
    if fields is None or len(fields) < 1:
        return False

    timespent = int(fields.get(TIMESPENT_FIELD_KEY, "0"))
    if timespent < 1:
        return False
    
    return True


def get_jira_base(url):
    """Extract JIRA repository base URL from dashboard, start page or similar URL"""
    return re.sub(r"((.*?)\:\/\/)|((\/)(.*?).jspa)|(\/secure)", "", url).strip("/")


def test_repos(potential_jira_repo_url_list, min_labeled_issue_count):
    """Test a list of URLs and return JIRA repository URLs with publicly available 'timespent' field and at least min_labeled_issue_count labeled issues
    A labeled issue is a resolved issue with 'timespent' reporten greater than zero.

    Arguments:\n
    potential_jira_repo_url_list -- a list containing potential JIRA repository URLs, such as ['https://jira.go2group.com/secure/Dashboard.jspa', 'not-jira-url.com', 'issues.apache.org/jira']\n
    min_labeled_issue_count -- the minimal number of labeled issues for a repository to be qualified
    """
    
    examined_website_count = len(potential_jira_repo_url_list)
    open_repos = []
    too_small_count = 0
    unreadable_labels_count = 0

    for url in potential_jira_repo_url_list:

        url = get_jira_base(url.strip())
        is_already_added = sum(1 for repo in open_repos if repo["url"] == url) > 0
        if is_already_added == True:
            continue
        print("-----------------------------")
        print("Trying out %s" % url)

        repository_search_url = get_repository_search_url(url)
        
        total_labeled_issues = get_issue_count(repository_search_url, None, LABELED_DATA_JQL)
        if total_labeled_issues < min_labeled_issue_count:
            if total_labeled_issues > 0:
                too_small_count = too_small_count + 1
            continue

        if not is_timespent_returned(repository_search_url):
            unreadable_labels_count = unreadable_labels_count + 1
            continue

        total_issues = get_issue_count(repository_search_url)
        labeling_coverage = total_labeled_issues / total_issues * 100 if total_issues > 0 else 0

        issue_statement = "%s contains %d issues in total of which %d (%.2f%%) are labeled."
        print(issue_statement % (url, total_issues, total_labeled_issues, labeling_coverage))

        repo = {
            "url": url,
            "labeled_issues": total_labeled_issues,
            "total_issues": total_issues,
            "labeling_coverage": round(labeling_coverage, 2)
        }
        open_repos.append(repo)

    open_repos = sorted(open_repos, key=lambda result: result["labeled_issues"], reverse=True)
    return {
        "examined_websites": examined_website_count,
        "too_small": too_small_count,
        "labels_unreadable": unreadable_labels_count,
        "open_repos": open_repos
    }

def print_test_result(result, min_labeled_issue_count):
    """Print JIRA repository test result in command line
    
    Arguments:\n
    result -- result dictionary\n
    min_labeled_issue_count -- the minimal number of labeled issues for a repository to be qualified, only for printing purposes
    """

    print("-----------------------------")
    print("Examined %d potential websites." % result["examined_websites"])
    if result["labels_unreadable"] > 0:
        print("%d were disqualified because 'timespent' field was not publicly accessible." % result["labels_unreadable"])
    if result["too_small"] > 0:
        print("%d were disqualified before they contained less than %d labeled issues." % (result["too_small"], min_labeled_issue_count))
    if len(result["open_repos"]) == 0:
        print("No repositories satisfying the criteria were found.")
        sys.exit()

    print("%d publicly available repositories fullfilling the criteria were found:" % len(result["open_repos"]))
    for repo in result["open_repos"]:
        print("%s - %d labeled issues, %d total issues, %.2f%% labeling coverage" % (repo["url"], repo["labeled_issues"], repo["total_issues"], repo["labeling_coverage"]))

if __name__ == "__main__":

    with open(POTENTIAL_REPOS_FILENAME) as file:
        potential_repo_url_list = file.readlines()

    if potential_repo_url_list is None or len(potential_repo_url_list) == 0:
        print("%s does not contain any potential JIRA repository URLs" % POTENTIAL_REPOS_FILENAME)
        sys.exit()
    
    min_labeled_issue_count = int(input("Please enter the minimum number of resolved issues with time spent greater than zero necessary to qualify a repository: "))
    
    result = test_repos(potential_repo_url_list, min_labeled_issue_count)
    print_test_result(result, min_labeled_issue_count)