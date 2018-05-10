import sys
import os

from fetch.fetch_data import fetch_data
from preprocess.clean_module import clean_text
from preprocess.merge_module import merge_data
from preprocess.filter_module import filter_data
from preprocess.filter_config import FilterConfig
from translate.tokens_module import count_tokens
from translate.dictionary_module import create_dictionary
from utilities.constants import *
from utilities.load_data import load_json

def get_clean_filename(dataset, labeling):
    return get_data_filename(dataset, labeling, CLEANED_POSTFIX, JSON_FILE_EXTENSION)

def create_training_dataset(configuration):

    repositories = configuration.get("repositories")
    if repositories is None:
        print("Configuration doesn't contain any repositories")
        return
        
    print("FETCHING DATA")
    for repository in repositories:
        if not os.path.exists("%s/%s" % (DATA_FOLDER, repository["key"])):
            try:
                fetch_data(repository["key"], repository["url"])
            except Exception as e:
                print("Skipping because the following exception was thrown:")
                print(e)
                continue
    
    print("PREPROCESSING DATA")
    repository_keys = [repository["key"] for repository in repositories]
    for repository_key in repository_keys:
        if not os.path.exists(get_clean_filename(repository_key, LABELED_FILENAME)) or not os.path.exists(get_clean_filename(repository_key, UNLABELED_FILENAME)):
            try:
                clean_text([repository_key])
            except Exception as e:
                print("Skipping because the following exception was thrown:")
                print(e)
                continue
    training_dataset_name = merge_data(repository_keys)

    return training_dataset_name

def run_configuration(configuration_name, training_dataset_name = "0"):

    filename = get_running_configuration_filename(configuration_name)
    configuration = load_json(filename)

    if training_dataset_name != "0":
        training_dataset_name = create_training_dataset(configuration)

    filter_config = FilterConfig()
    filter_params = configuration.get("filter")
    if filter_params is not None:
        for param_name, param_value in filter_params.items():
            filter_config.set_param(param_name, param_value)
    filter_data(training_dataset_name, filter_config)
    
    pretrain_config = configuration.get("pretrain")
    if pretrain_config is not None:
        print("PREPARING DATA FOR PRETRAINING - CONVERTING WORDS TO INTEGERS")
        
        count_tokens(training_dataset_name)
        create_dictionary(training_dataset_name, TOTAL_KEY, pretrain_config.get("min_word_occurence"))

if len(sys.argv) < 2:
    print("Please enter the name of the configuration which you want to run")

if len(sys.argv) == 3:
    run_configuration(sys.argv[1], sys.argv[2])

run_configuration(sys.argv[1])

