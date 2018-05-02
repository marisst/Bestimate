import sys

from fetching.fetch_data import fetch_data
from utilities.constants import *
from utilities.load_data import load_json

def run_configuration(configuration_name):

    filename = get_running_configuration_filename(configuration_name)
    configuration = load_json(filename)

    repositories = configuration.get("repositories")
    if repositories is not None:
        for repository in repositories:
            if not os.path.exists(DATA_FOLDER)
        

if len(sys.argv) != 2:
    print("Please enter the name of the configuration which you want to run")


