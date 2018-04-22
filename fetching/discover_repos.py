import json
import re
import requests
import sys

from fetching.count_issues import count as get_issue_count
from utilities.constants import *
from utilities.string_utils import get_part_strings

BING_SEARCH_URL = "https://api.cognitive.microsoft.com/bing/v7.0/search"
KEYWORDS = [ "intitle:\"system dashboard\"" ]
MIN_LABELED_ISSUE_COUNT = 100

def bing_search(query, bing_api_key):

    PAGE_SIZE = 50
    results = []
    totalEstimatedMatches = 0
    page = 0
    
    while page * PAGE_SIZE <= totalEstimatedMatches:

        payload = {'q': query, 'count': PAGE_SIZE, 'offset': page * PAGE_SIZE, 'responseFilter': 'Webpages'}
        headers = {'Ocp-Apim-Subscription-Key': bing_api_key}

        page = page + 1

        try:
            response = requests.get(BING_SEARCH_URL, params=payload, headers=headers)
        except requests.exceptions.RequestException:
            print("Request exception, jump over page")
            continue

        if response.status_code != 200:
            print("Unsuccessful status code %d, jump over page" % response.status_code)
            continue

        try:
            json_response = response.json()
        except json.JSONDecodeError:
            print("Could not decode response, jump over page")
            continue

        totalEstimatedMatches = json_response.get("webPages").get("totalEstimatedMatches")

        webpages = json_response.get("webPages").get("value")
        result = [webpage.get("url") for webpage in webpages]
        results = results + result
        print("%d (%.2f%%) of %d potential JIRA repository links retrieved" % get_part_strings(len(results), totalEstimatedMatches))
        
    return results

def get_jira_base(url):

    return re.sub(r"((.*?)\:\/\/)|((\/)(.*?).jspa)|(\/secure)", "", url).strip("/")

def is_timespent_returned(repository_search_url):
    
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

def discover_repositories(bing_api_key):

    search_result_urls = set()
    for keyword in KEYWORDS:
        results = bing_search(keyword, bing_api_key)
        search_result_urls = search_result_urls.union(results)
    
    examined_website_count = len(search_result_urls)
    open_repos = []
    too_small_count = 0
    unreadable_labels_count = 0
    for url in search_result_urls:

        base_url = get_jira_base(url)
        print("-----------------------------")
        print("Trying out %s" % base_url)

        repository_search_url = get_repository_search_url(base_url)
        
        total_labeled_issues = get_issue_count(repository_search_url, None, LABELED_DATA_JQL)
        if total_labeled_issues < MIN_LABELED_ISSUE_COUNT:
            if total_labeled_issues > 0:
                too_small_count = too_small_count + 1
            continue

        if not is_timespent_returned(repository_search_url):
            unreadable_labels_count = unreadable_labels_count + 1
            continue

        total_issues = get_issue_count(repository_search_url)
        labeling_coverage = total_labeled_issues / total_issues * 100 if total_issues > 0 else 0

        issue_statement = "%s contains %d issues in total of which %d (%.2f%%) are labeled."
        print(issue_statement % (base_url, total_issues, total_labeled_issues, labeling_coverage))

        repo = {
            "url": base_url,
            "labeled_issues": total_labeled_issues,
            "total_issues": total_issues,
            "labeling_coverage": round(labeling_coverage, 2)
        }
        print(repo)
        open_repos.append(repo)

    open_repos = sorted(open_repos, key=lambda result: result["labeled_issues"], reverse=True)
    results = {
        "examined_websites": examined_website_count,
        "too_small": too_small_count,
        "labels_unreadable": unreadable_labels_count,
        "open_repos": open_repos
    }
    

    filename = get_repo_list_filename()
    with open(filename, 'w') as file:
        json.dump(results, file, indent=JSON_INDENT)
    print("Search finished and results saved at %s" % filename)

sys_argv_count = len(sys.argv)

if sys_argv_count == 2:
    discover_repositories(sys.argv[1])
    sys.exit()

print("Please pass your Bing Web Search API key (see https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/)")