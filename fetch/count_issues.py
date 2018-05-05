import json
import requests

from utilities.constants import *

def count(repository_search_url, auth=None, jql=""):
    
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