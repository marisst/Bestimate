import sys
import os

from fetch.fetch_data import fetch_data
from preprocess.clean_module import clean_text
from utilities.constants import *
from utilities.load_data import load_json

def get_clean_filename(dataset, labeling):
    return get_data_filename(dataset, labeling, CLEANED_POSTFIX, JSON_FILE_EXTENSION)

def run_configuration(configuration_name):

    filename = get_running_configuration_filename(configuration_name)
    configuration = load_json(filename)

    repositories = configuration.get("repositories")
    if repositories is not None:
        
        # fetch data
        for repository in repositories:
            if not os.path.exists("%s/%s" % (DATA_FOLDER, repository["key"])):
                try:
                    fetch_data(repository["key"], repository["url"])
                except Exception as e:
                    print("Skipping because the following exception was thrown:")
                    print(e)
                    continue
        
        # preprocess
        repository_keys = [repository["key"] for repository in repositories]
        for repository_key in repository_keys:
            if not os.path.exists(get_clean_filename(repository_key, LABELED_FILENAME)) or not os.path.exists(get_clean_filename(repository_key, UNLABELED_FILENAME)):
                try:
                    clean_text([repository_key])
                except Exception as e:
                    print("Skipping because the following exception was thrown:")
                    print(e)
                    continue



if len(sys.argv) != 2:
    print("Please enter the name of the configuration which you want to run")

run_configuration(sys.argv[1])

