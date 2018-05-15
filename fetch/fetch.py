import sys
import os

from fetch.fetch_data import fetch_data
from preprocess.clean_module import clean_text
from preprocess.merge_module import merge_data
from preprocess.filter_module import filter_data
from preprocess.filter_config import FilterConfig
from translate.tokens_module import count_tokens
from translate.dictionary_module import create_dictionary
from pretrain.train_gensim import train_gensim
from utilities.constants import *
from utilities.load_data import load_json


def get_clean_filename(dataset, labeling):
    return get_data_filename(dataset, labeling, CLEANED_POSTFIX, JSON_FILE_EXTENSION)


def create_training_dataset(repositories):

    if repositories is None:
        print("Configuration doesn't contain any repositories")
        return
        
    print("FETCHING DATA")
    for repository in repositories:
        if not os.path.exists("%s/%s" % (DATA_FOLDER, repository[0])):
            try:
                fetch_data(repository[0], repository[1])
            except Exception as e:
                print("Skipping because the following exception was thrown:")
                print(e)
                continue
    
    print("CLEANING DATA")
    repository_keys = [repository["key"] for repository in repositories]
    for repository_key in repository_keys:
        if not os.path.exists(get_clean_filename(repository_key, LABELED_FILENAME)) or not os.path.exists(get_clean_filename(repository_key, UNLABELED_FILENAME)):
            try:
                clean_text([repository_key])
            except Exception as e:
                print("Skipping because the following exception was thrown:")
                print(e)
                continue


def run_configuration():

    repositories = load_json("fetch/datasets.json")
    create_training_dataset(repositories)

