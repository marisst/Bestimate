from hyperopt import fmin, tpe, hp, STATUS_FAIL, STATUS_OK
import json

from data_preprocessing.filter_config import FilterConfig
from data_preprocessing.filter_data import filter_data
from embedding_pretraining.count_tokens import count_tokens
from embedding_pretraining.spacy_lookup import spacy_lookup
from embedding_pretraining.train_gensim import train_gensim
from training.train import train_on_dataset
from utilities.constants import *
from utilities.file_utils import load_json, get_next_subfolder_name, create_subfolder

space = {
    'min_word_count': hp.choice('min_word_count', [1, 4, 8, 16, 32, 64]),
    'min_timespent_minutes': hp.choice('min_timespent_minutes', [1, 4, 8, 16, 32]),
    'max_timespent_minutes': hp.choice('max_timespent_minutes', [240, 360, 480, 720, 960, 1920]),
    'min_project_size': hp.choice('min_project_size', [1, 32, 64, 128, 512, 1024]),
    'even_distribution': hp.choice('even_distribution', [True, False]),
    'word_embeddings': hp.choice('word_embeddings', [
        {
            "type": "spacy"
        },
        {
            "type": "gensim",
            "algorithm": hp.choice("word_embeddings_algorithm", ["skip-gram", "CBOW"]),
            "embedding_size": hp.choice("word_embeddings_embedding_size", [5, 10, 50, 100, 200, 300]),
            "minimum_count": hp.choice("word_embeddings_minimum_count", [1, 2, 4, 8, 16]),
            "window_size": hp.choice("word_embeddings_window_size", [3, 5, 7, 9, 11, 15]),
            "iterations": hp.choice("word_embeddings_iterations", [1, 3, 5, 7, 9])
        }
    ]),
    'model_params':
    {
        'max_words': hp.choice('max_words', [32, 64, 128, 256]),
        'lstm_node_count': hp.choice('lstm_node_count', [4, 8, 16, 32, 64, 128, 256, 512]),
        'lstm_recurrent_dropout': hp.uniform('lstm_recurrent_dropout', 0, 0.7),
        'lstm_dropout': hp.uniform('lstm_dropout', 0, 0.7),
        'highway_layer_count': hp.choice('highway_layer_count', [4, 8, 16, 32, 64, 128]),
        'highway_activation': hp.choice('highway_activation', ['relu', 'tanh']),
        'dropout': hp.uniform('dropout', 0, 0.7),
        'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256]),
        'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
        'loss': hp.choice('loss', ['mean_squared_error', 'mean_absolute_error']) #mean_absolute_percentage_error
    }
}

def objective(configuration):

    print("--- NEW CONFIGURATION ---")
    print(configuration)

    training_dataset_name = configuration['training_dataset_id']
    training_session_id = configuration['training_session_id']

    training_session_folder = "%s/%s" % (RESULTS_FOLDER, training_session_id)
    run_id = get_next_subfolder_name(training_session_folder)
    create_subfolder(training_session_folder, run_id)

    notes_filename = "%s/%s/notes.txt" % (training_session_folder, run_id)
    with open(notes_filename, "a") as notes_file:
        print(json.dumps(configuration, indent=JSON_INDENT), file=notes_file)

    filter_config = FilterConfig()
    filter_config.min_word_count = configuration["min_word_count"]
    filter_config.min_timespent_minutes = configuration["min_timespent_minutes"]
    filter_config.max_timespent_minutes = configuration["max_timespent_minutes"]
    filter_config.min_project_size = configuration["min_project_size"]
    filter_config.even_distribution_bin_count = 5 if configuration["even_distribution"] == True else 0
    filtered_datapoint_count = filter_data(training_dataset_name, filter_config, notes_filename)
    if filtered_datapoint_count == 0:
        return {
            "status": STATUS_FAIL
        }

    
    count_tokens(training_dataset_name, notes_filename)
    emb_config = configuration["word_embeddings"]
    if emb_config["type"] == "spacy":
        spacy_lookup(training_dataset_name, notes_filename)
    
    if emb_config["type"] == "gensim":
        train_gensim(
            training_dataset_name,
            emb_config["algorithm"],
            emb_config["embedding_size"],
            emb_config["minimum_count"],
            emb_config["window_size"], 
            emb_config["iterations"],
            notes_filename)

    loss = train_on_dataset(training_dataset_name, emb_config["type"], configuration["model_params"], notes_filename, session_id=training_session_id, run_id=run_id)
    return {
        "loss": loss,
        "status": STATUS_OK
    }

def optimize_model(training_dataset_id):

    space["training_dataset_id"] = training_dataset_id
    space["training_session_id"] = get_next_subfolder_name(RESULTS_FOLDER)
    create_subfolder(RESULTS_FOLDER, space["training_session_id"])

    best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=100)

    print("BEST:")
    print(best)

if __name__ == "__main__":

    training_dataset_id = input("Please enter the identifier of the dataset you want train the model on: ")
    optimize_model(training_dataset_id)
