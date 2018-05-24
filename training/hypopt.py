from hyperopt import fmin, tpe, hp, STATUS_FAIL, STATUS_OK
from hyperopt.pyll.base import scope
import gc
import json
import numpy as np
import sys

from data_preprocessing.filter_config import FilterConfig
from data_preprocessing.filter_data import filter_data
from embedding_pretraining.count_tokens import count_tokens
from embedding_pretraining.spacy_lookup import spacy_lookup
from embedding_pretraining.train_gensim import train_gensim
from training.train import train_on_dataset
from utilities.constants import *
from utilities.file_utils import load_json, get_next_subfolder_name, create_subfolder


def create_space(embedding_type):

    if embedding_type == "spacy":
        embedding_space = {
                "type": "spacy"
            }

    if embedding_type == "gensim":
        embedding_space = {
                "type": "gensim",
                "algorithm": hp.choice("word_embeddings_algorithm", ["skip-gram", "CBOW"]),
                "embedding_size": scope.int(hp.quniform("word_embeddings_embedding_size", 5, 500, 1)),
                "minimum_count": scope.int(hp.quniform("word_embeddings_minimum_count", 1, 15, 1)),
                "window_size": scope.int(hp.qnormal("word_embeddings_window_size", 7, 3, 1)),
                "iterations": scope.int(hp.qnormal("word_embeddings_iterations", 5, 3, 1))
            }

    space = {
        'min_word_count': scope.int(hp.qnormal('min_word_count', 15, 4, 1)),
        'min_timespent_minutes': 10,
        'max_timespent_minutes': 960,
        'min_project_size': hp.choice("min_project_size", [1, 20, 50, 200, 500]),
        'even_distribution': False,
        'word_embeddings': embedding_space,
        'model_params':
        {
            'max_words': scope.int(hp.qnormal('max_words', 120, 20, 1)),
            'lstm_node_count': scope.int(hp.quniform('lstm_node_count', 5, 150, 1)),
            'lstm_recurrent_dropout': hp.uniform('lstm_recurrent_dropout', 0, 0.7),
            'lstm_dropout': hp.uniform('lstm_dropout', 0, 0.7),
            'highway_layer_count': scope.int(hp.quniform('highway_layer_count', 5, 150, 1)),
            'highway_activation': hp.choice("highway_activation", ["relu", "tanh"]),
            'dropout': hp.uniform('dropout', 0, 0.7),
            'batch_size': 200,
            'optimizer': hp.choice('optimizer', [
                ('rmsprop', hp.uniform('rmsprop_lr', 0.0005, 0.005)),
                ('adam', hp.uniform('adam_lr', 0.0005, 0.005)),
                ('sgd', hp.uniform('sgd_lr', 0.005, 0.05))]),
            'loss': 'mean_absolute_error'
        }
    }
    
    for regularizer in REGULARIZERS:
        regularizer_name = "%s-regularizer-" % regularizer
        for regularizer_type in ["l1", "l2"]:
            space['model_params'][regularizer_name + regularizer_type] = hp.choice(regularizer_name + regularizer_type, [
                (True, hp.uniform(regularizer_name + regularizer_type + '-constant', 0, 0.001)),
                (False, None)
            ])

    return space


def remove_negative_values(nested_dictionary):

    result = {}
    for key, value in nested_dictionary.items():
        if isinstance(value, dict):
            result[key] = remove_negative_values(value)
        elif isinstance(value, int):
            result[key] = max(0, value)
        else:
            result[key] = value
    return result


def objective(configuration):

    print("--- NEW CONFIGURATION ---")

    configuration = remove_negative_values(configuration)
    if configuration["model_params"]["max_words"] == 0:
        return {
            "status": STATUS_FAIL
        }

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
    labeled_data, unlabeled_data = filter_data(training_dataset_name, filter_config, notes_filename, save=False)

    emb_config = configuration["word_embeddings"]
    if emb_config["type"] == "spacy":
        unlabeled_data = None
        gc.collect()
    
    if labeled_data is None or len(labeled_data) == 0:
        return {
            "status": STATUS_FAIL
        }

    data = labeled_data
    if unlabeled_data is not None:
        data = data + unlabeled_data
    
    gensim_model = None
    if emb_config["type"] == "gensim":
        gensim_model = train_gensim(
            training_dataset_name,
            emb_config["algorithm"],
            emb_config["embedding_size"],
            emb_config["minimum_count"],
            emb_config["window_size"], 
            emb_config["iterations"],
            notes_filename,
            data=data,
            save=False)

    loss, val_loss = train_on_dataset(
        training_dataset_name,
        emb_config["type"],
        configuration,
        notes_filename,
        session_id=training_session_id,
        run_id=run_id,
        labeled_data=labeled_data,
        gensim_model=gensim_model)

    data = None
    unlabeled_data = None
    labeled_data = None
    gc.collect()

    log_filename = "%s/%s/%s%s" % (RESULTS_FOLDER, training_session_id, RESULTS_FILENAME, TEXT_FILE_EXTENSION)
    with open(log_filename, "a") as log_file:
        print("Run: %s, Loss: %.4f, val_loss: %.4f" % (run_id, loss, val_loss), file=log_file)

    return {
        "loss": loss,
        "val_loss": val_loss,
        "status": STATUS_OK
    }


def optimize_model(training_dataset_id, embedding_type):

    space = create_space(embedding_type)

    space["training_dataset_id"] = training_dataset_id
    space["training_session_id"] = "%s_%s_%s" % (get_next_subfolder_name(RESULTS_FOLDER), training_dataset_id, embedding_type)
    create_subfolder(RESULTS_FOLDER, space["training_session_id"])

    evals = 150 if embedding_type == "spacy" else 200

    best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=evals,
    rstate=np.random.RandomState(7),
    )

    print("BEST:")
    print(best)

if __name__ == "__main__":
    optimize_model(sys.argv[1], sys.argv[2])