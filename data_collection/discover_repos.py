from googleapiclient.discovery import build
import json
import math
import requests
import sys
import time

from data_collection.count_issues import count as get_issue_count
from data_collection.test_repos import discover_repositories, get_jira_base
from utilities.constants import *
from utilities.string_utils import get_part_strings

BING_SEARCH_URL = "https://api.cognitive.microsoft.com/bing/v7.0/search"
BING_KEYWORDS = [ "intitle:\"system dashboard\"" ]
GOOGLE_KEYWORDS = ["intitle:System Dashboard - JIRA", "intitle:\"System Dashboard - JIRA\"", "allintitle:\"system dashboard\""]
MIN_LABELED_ISSUE_COUNT = 100

def bing_search(query, bing_api_key):

    PAGE_SIZE = 50
    results = set()
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
    
        webpages = json_response.get("webPages").get("value")
        result = set([get_jira_base(webpage.get("url")) for webpage in webpages])
        results = results.union(result)

        totalEstimatedMatches = json_response.get("webPages").get("totalEstimatedMatches")
        totalPages = math.ceil(totalEstimatedMatches / PAGE_SIZE)

        print("%d (%.2f%%) of %d result pages processed" % get_part_strings(page, totalPages))
    
    print("%d potential JIRA instances found" % len(results))
    return results

def google_search(keyword, google_api_key, cse_id):

    PAGE_SIZE = 10

    service = build("customsearch", "v1", developerKey=google_api_key)
    results = set()
    total_results = 0
    page = 0

    while page * PAGE_SIZE <= total_results:

        if page % 5 == 0:
            time.sleep(5)

        try:
            res = service.cse().list(q=keyword, cx=cse_id, num=PAGE_SIZE, start=page * PAGE_SIZE + 1).execute()
            page = page + 1
            total_results = int(res.get("searchInformation").get("totalResults"))
        except Exception as e:
            print("Exception, skip page")
            print(e)
            break

        try:
            items = res.get('items')
        except:
            continue

        for item in items:
            try:
                url = get_jira_base(item.get("link"))
            except:
                continue
            results.add(url)

        totalPages = math.ceil(total_results / PAGE_SIZE)
        print("%d (%.2f%%) of %d result pages processed" % get_part_strings(page, totalPages))

    print("%d potential JIRA instances found" % len(results))
    return results


def discover_repositories_bing(bing_api_key):

    search_result_urls = set()
    for keyword in BING_KEYWORDS:
        results = bing_search(keyword, bing_api_key)
        search_result_urls = search_result_urls.union(results)

    results = discover_repositories(search_result_urls)

    filename = get_repo_list_filename(BING)
    with open(filename, 'w') as file:
        json.dump(results, file, indent=JSON_INDENT)
    print("Search finished and results saved at %s" % filename)

def discover_repositories_google(google_api_key, cse_id):

    search_result_urls = set()
    for keyword in GOOGLE_KEYWORDS:
        results = google_search(keyword, google_api_key, cse_id)
        search_result_urls = search_result_urls.union(results)

    results = discover_repositories(search_result_urls)

    filename = get_repo_list_filename(GOOGLE)
    with open(filename, 'w') as file:
        json.dump(results, file, indent=JSON_INDENT)
    print("Search finished and results saved at %s" % filename)

search_engine = input("Please enter the name of the search engine you want to use (google or bing): ").lower()
if search_engine not in SEARCH_ENGINES:
    print("Please choose one of the following search engines:", *SEARCH_ENGINES)
    sys.exit()

if search_engine == BING:
    print("Note: You can get Bing Web Search API key from https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/")
    bing_api_key = input("Bing Web Search API key: ")
    discover_repositories_bing(bing_api_key)

if search_engine == GOOGLE:
    print("Note: You can configute and get Google Custom Search Engine id and API key as described on https://stackoverflow.com/a/37084643 (step 1 and 2). The custom search engine should be configured to search the whole web.")
    google_api_key = input("Google API key: ")
    cse_id = input("Google Custom Search Engine ID: ")
    discover_repositories_google(google_api_key, cse_id)