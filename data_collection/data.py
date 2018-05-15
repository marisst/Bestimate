import sys
from fetch.fetch_data import fetch_data

def get_auth():

    authorize = input("Do you want to sign in? (y/n) ") == "y"

    if authorize:
        username = input("Username: ")
        api_token = input("API token: ")

    return (username, api_token) if authorize is True else None

sys_argv_count = len(sys.argv)

if sys_argv_count == 3:
    auth = get_auth()
    fetch_data(sys.argv[1], sys.argv[2], auth)
    sys.exit()

print("Please pass 2 arguments to this script:\n1. Dataset name that will be used as folder name\n2. JIRA repository URL")
print("Example request: python fetch_data.py dataset_name jira.repositoryname.com")