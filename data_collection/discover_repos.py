from googleapiclient.discovery import build
import json
import math
import requests
import sys
import time

from data_collection.test_repos import test_repos, print_test_result
from utilities.string_utils import get_part_strings

BING = "bing"
GOOGLE = "google"
SEARCH_ENGINES = [BING, GOOGLE]
BING_SEARCH_URL = "https://api.cognitive.microsoft.com/bing/v7.0/search"
BING_KEYWORDS = [ "intitle:\"system dashboard\"" ]
GOOGLE_KEYWORDS = ["intitle:System Dashboard - JIRA", "intitle:\"System Dashboard - JIRA\"", "allintitle:\"system dashboard\""]

def bing_search(query, bing_api_key):
    """Returns a list of URLs obtained in a Bing search query using Bing Web Search API v7 https://docs.microsoft.com/en-gb/rest/api/cognitiveservices/bing-web-api-v7-reference
    
    Arguments:\n
    query -- Bing search query, advanced operators may be used, see https://msdn.microsoft.com/library/ff795620.aspx\n
    bing_api_key -- Bing Web Search API key which can be obtained at https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/
    """

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
            print("Could not decode response or it didn't contain an URL, jump over page")
            continue

        if json_response == None or json_response.get("webPages") == None or json_response.get("webPages").get("value") == None:
            print("Couldn't get web pages from response")
            continue

        webpages = json_response.get("webPages").get("value")
        result = set([webpage.get("url") for webpage in webpages if webpage.get("url") != None])
        
        results = results.union(result)

        totalEstimatedMatches = json_response.get("webPages").get("totalEstimatedMatches")
        totalPages = math.ceil(totalEstimatedMatches / PAGE_SIZE)

        print("%d (%.2f%%) of %d result pages processed" % get_part_strings(page, totalPages))
    
    print("%d potential JIRA instances found" % len(results))
    return results


def google_search(keyword, google_api_key, cse_id):
    """Returns a list of URLs obtained in a Google search query using Google Custom Search Engine API https://developers.google.com/custom-search/json-api/v1/reference/cse/list
    The Custom Search Engine instance should be configured to search the whole Web as described in the first two steps at https://stackoverflow.com/a/37084643

    Arguments:\n
    keyword -- Google search query, advanced operators may be used, see https://bynd.com/news-ideas/google-advanced-search-comprehensive-list-google-search-operators/\n
    google_api_key -- Google API key, see https://developers.google.com/api-client-library/python/guide/aaa_apikeys\n
    cse_id -- Custom Search Engine ID, see https://cse.google.com/cse/
    """

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
                url = item.get("link")
            except:
                continue
            results.add(url)

        totalPages = math.ceil(total_results / PAGE_SIZE)
        print("%d (%.2f%%) of %d result pages processed" % get_part_strings(page, totalPages))

    print("%d potential JIRA instances found" % len(results))
    return results


if __name__ == "__main__":

    min_labeled_issue_count = int(input("Please enter the minimum number of resolved issues with time spent greater than zero necessary to qualify a repository: "))
    search_result_urls = set()
    search_engine = input("Please enter the name of the search engine you want to use (google or bing): ").lower()
    if search_engine not in SEARCH_ENGINES:
        print("Please choose one of the following search engines:", *SEARCH_ENGINES)
        sys.exit()

    if search_engine == BING:
        print("Note: You can get Bing Web Search API key from https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/")
        bing_api_key = input("Bing Web Search API key: ")
        for keyword in BING_KEYWORDS:
            print("Searching by keyword:", keyword)
            results = bing_search(keyword, bing_api_key)
            search_result_urls = search_result_urls.union(results)

    if search_engine == GOOGLE:
        print("Note: You can obtain Google API key and Google Custom Search Engine ID as described on https://stackoverflow.com/a/37084643 (step 1 and 2). The custom search engine should be configured to search the whole web.")
        google_api_key = input("Google API key: ")
        cse_id = input("Google Custom Search Engine ID: ")
        for keyword in GOOGLE_KEYWORDS:
            print("Searching by keyword:", keyword)
            results = google_search(keyword, google_api_key, cse_id)
            search_result_urls = search_result_urls.union(results)
            
    result = test_repos(search_result_urls, min_labeled_issue_count)
    print_test_result(result, min_labeled_issue_count)