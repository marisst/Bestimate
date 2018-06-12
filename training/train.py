import matplotlib
matplotlib.use('Agg')

from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from keras.losses import mean_squared_error, mean_absolute_error
from keras.optimizers import RMSprop, Adam, SGD
from keras.models import load_model
from keras import backend as K
import json
import matplotlib.pyplot as plt
import numpy as np
import sys
from gensim.models import Word2Vec
import spacy
from functools import partial
import os
import multiprocessing

from embedding_pretraining.train_gensim import train_gensim
from training import calculate_baselines as bsl
from training import load_data as load
from training import model as mdl
from training import save_results as save
from training.data_generator import DataGenerator
from training.graph_helpers import plot_losses, create_prediction_scatter
from utilities.constants import *
from utilities.file_utils import create_subfolder, get_next_subfolder_name
from utilities.input_parser import select_from_list

# training parameters
max_epochs = 1000
split_percentages = 60, 20


def spacy_lookup(nlp, word):

    doc = nlp(word)
    if doc.has_vector == False:
        return None

    if not np.any(doc[0].vector):
        return None

    return doc[0].vector.tolist()


def gensim_lookup(word_vectors, word):

    if word not in word_vectors:
        return None
    
    return word_vectors.get_vector(word)


def calculate_validation_result(model, x_valid, y_valid, y_train, loss_function, model_params, vector_dictionary, notes_filename, fixed_generator_error):

    validation_generator = DataGenerator(
        x_valid,
        y_valid,
        model_params["batch_size"] if fixed_generator_error == False else len(y_valid),
        True if model_params["lstm_count"] == 2 else False,
        model_params["max_words"],
        vector_dictionary)
    validation_loss = model.evaluate_generator(generator=validation_generator, use_multiprocessing=True, workers=model_params["workers"])

    mean_baseline = loss_function(y_valid, np.mean(y_train))
    median_baseline = loss_function(y_valid, np.median(y_train))
    human_loss = bsl.mean_human_absolute_error(y_valid)
    validation_result = validation_loss / min([mean_baseline, median_baseline])
    human_score = (min([mean_baseline, median_baseline]) - validation_loss) / (min([mean_baseline, median_baseline]) - human_loss) * 100

    with open(notes_filename, "a") as notes_file:
        print("Human loss (valid):", human_loss, file=notes_file)
        print("Mean loss (valid):", mean_baseline, file=notes_file)
        print("Median loss (valid):", median_baseline, file=notes_file)
        print("Validation result:", validation_result, file=notes_file)
        print("Human score (valid):", human_score, file=notes_file)

    return validation_result


def train_on_dataset(params, labeled_data=None, generate_graphs = False):

    # this is because of a bug in generator and reproducibility
    fixed_generator_error = generate_graphs == True    

    if params.get("training_session_id") == None:
        params["training_session_id"] = "%s_%s_%s" % (get_next_subfolder_name(RESULTS_FOLDER), params["training_dataset_id"], params["word_embeddings"]["type"])
        create_subfolder(RESULTS_FOLDER, params["training_session_id"])

    run_folder = "%s/%s/%s" % (RESULTS_FOLDER, params["training_session_id"], params["run_id"])
    if os.path.exists(run_folder) == False:
        create_subfolder("%s/%s" % (RESULTS_FOLDER, params["training_session_id"]), params["run_id"])

    notes_filename = "%s/%s/%s/notes.txt" % (RESULTS_FOLDER, params["training_session_id"], params["run_id"])
    with open(notes_filename, "a") as notes_file:
        print(json.dumps(params, indent=JSON_INDENT), file=notes_file) 

    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    K.set_session(K.tf.Session(config=config))

    embedding_type = params["word_embeddings"]["type"]
    if embedding_type == "spacy":
        nlp = spacy.load('en_vectors_web_lg', disable=['parser', 'tagger', 'entity'])
        lookup = partial(spacy_lookup, nlp)

    if embedding_type == "gensim":
        model_filename = get_dataset_filename(params["training_dataset_id"], ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
        if os.path.exists(model_filename):
            gensim_model = Word2Vec.load(model_filename)
        else:
            gensim_model = train_gensim(
                params["training_dataset_id"],
                params["word_embeddings"]["algorithm"],
                params["word_embeddings"]["embedding_size"],
                params["word_embeddings"]["minimum_count"],
                params["word_embeddings"]["window_size"], 
                params["word_embeddings"]["iterations"],
                notes_filename,
                save=False,
                workers=params["model_params"]["workers"])
        lookup = partial(gensim_lookup, gensim_model.wv)
        del gensim_model

    # load and arrange data
    model_params = params["model_params"]
    data, vector_dictionary = load.load_and_arrange(
        params["training_dataset_id"],
        split_percentages,
        True if model_params["lstm_count"] == 2 else False,
        model_params["max_words"],
        lookup,
        labeled_data=labeled_data)
    del labeled_data
    x_train, y_train, x_test, y_test, x_valid, y_valid = data
    
    if model_params["loss"] == "mean_squared_error":
        loss_function = bsl.mean_squared_error

    if model_params["loss"] == "mean_absolute_error":
        loss_function = bsl.mean_absolute_error
        human_loss = bsl.mean_human_absolute_error(y_test)
        with open(notes_filename, "a") as notes_file:
            print("Human loss (test):", human_loss, file=notes_file)

    if model_params["loss"] == "mean_absolute_percentage_error":
        loss_function = bsl.mean_absolute_percentage_error

    # calculate baseline losses
    mean_baseline = loss_function(y_test, np.mean(y_train))
    median_baseline = loss_function(y_test, np.median(y_train))
    with open(notes_filename, "a") as notes_file:
        print("Mean loss (test):", mean_baseline, file=notes_file)
        print("Median loss (test):", median_baseline, file=notes_file)

    # create model
    embedding_size = vector_dictionary.shape[1]
    model = mdl.create_model(model_params["max_words"], embedding_size, model_params)

    if model_params["optimizer"][0] == 'rmsprop':
        optimizer = RMSprop(lr=model_params["optimizer"][1])
    elif model_params["optimizer"][0] == 'adam':
        optimizer = Adam(lr=model_params["optimizer"][1])
    elif model_params["optimizer"][0] == "sgd":
        optimizer = SGD(lr=model_params["optimizer"][1])

    model.compile(loss=model_params["loss"], optimizer=optimizer)

    # create results files
    weigths_directory_name = "%s/%s/%s" % (RESULTS_FOLDER, params["training_session_id"], params["run_id"])
    #weigths_filename = get_weigths_filename(params["training_dataset_id"], training_session_name)
    #save_weights = ModelCheckpoint(weigths_filename)
    results_filename = "%s/%s%s" % (weigths_directory_name, RESULTS_FILENAME, TEXT_FILE_EXTENSION)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(results_filename, epoch, logs))

    best_model_filename = weigths_directory_name + "/model.h5"
    save_best_model = ModelCheckpoint(best_model_filename, save_best_only=True)

    training_generator = DataGenerator(
        x_train,
        y_train,
        model_params["batch_size"],
        True if model_params["lstm_count"] == 2 else False,
        model_params["max_words"],
        vector_dictionary)
    test_generator = DataGenerator(
        x_test,
        y_test,
        model_params["batch_size"] if fixed_generator_error == False else len(y_test),
        True if model_params["lstm_count"] == 2 else False,
        model_params["max_words"],
        vector_dictionary)

    # train and validate
    callbacks = [save_results, save_best_model, EarlyStopping(min_delta=params["min_delta"], patience=params["patience"])]
    # this sometimes throws OSError 35 on MAC OS X, https://github.com/urllib3/urllib3/issues/63
    history = model.fit_generator(
        generator = training_generator,
        validation_data = test_generator,
        use_multiprocessing=True,
        workers=model_params["workers"],
        callbacks=callbacks,
        epochs=max_epochs)

    del model
    K.clear_session()

    result = min(history.history["val_loss"]) / min([mean_baseline, median_baseline])
    with open(notes_filename, "a") as notes_file:
        print("Test result:", result, file=notes_file)

    best_model = load_model(best_model_filename)
    if generate_graphs == True:
        loss_plot_filename = "%s/%s%s" % (weigths_directory_name, "losses", PNG_FILE_XTENSION)
        train_baseline = min([loss_function(y_train, np.mean(y_train)), loss_function(y_train, np.median(y_train))])
        test_baseline = min([mean_baseline, median_baseline])
        plot_losses(history.history["loss"], history.history["val_loss"], train_baseline, test_baseline, model_params["loss"], loss_plot_filename)
        for x, y, filename, title in [
            (x_train, y_train, "%s/%s%s" % (weigths_directory_name, "train_pred", PNG_FILE_XTENSION), "Training dataset predictions"),
            (x_test, y_test, "%s/%s%s" % (weigths_directory_name, "test_pred", PNG_FILE_XTENSION), "Testing dataset predictions"),
            (x_valid, y_valid, "%s/%s%s" % (weigths_directory_name, "val_pred", PNG_FILE_XTENSION), "Validation dataset predictions")]:
            try:
                create_prediction_scatter(best_model, x, y, filename, title, model_params, vector_dictionary)
            except multiprocessing.pool.MaybeEncodingError:
                # ON OS X when data larger than 4 GB
                continue
            except OverflowError:
                continue

    val_result = calculate_validation_result(best_model, x_valid, y_valid, y_train, loss_function, model_params, vector_dictionary, notes_filename, fixed_generator_error)    
    os.remove(best_model_filename)

    return result, val_result


if __name__ == "__main__":

    dataset = input("Dataset (e.g. exo): ")
    
    embedding_config = {
        "type" : select_from_list(
            "Please choose embedding type",
            ["gensim", "spacy"])
    }
    if embedding_config["type"] == "gensim":
        embedding_config["algorithm"] = select_from_list(
            "Please select word embedding pretraining algorithm",
            ["skip-gram", "CBOW"])
        embedding_config["embedding_size"] = int(input("Embedding size: "))
        embedding_config["minimum_count"] = int(input("Minimum count: "))
        embedding_config["window_size"] = int(input("Window size: "))
        embedding_config["iterations"] = int(input("Iterations: "))

    model_config = {
        "lstm_count" : select_from_list(
            "Please select context model type", [
                "simple LSTM",
                "two separate LSTMS for summary and description",
                "bidirectional LSTM"], return_option_indexes=True),
        "lstm_node_count" : int(input("LSTM node count: ")),
        "conform_type" : select_from_list(
            "Select context transformation network type",
            ["hway", "dense"]),
        "conform_layer_count" : int(input("Context transformation network layer count: ")),
        "conform_activation" : select_from_list(
            "Please select context transformation network activation type",
            ["relu", "tanh"]),
        "dropout" : float(input("Final dropout: ")),
        "batch_size": 512,
        "loss": "mean_absolute_error",
        "workers": int(input("Workers: ")),
        "optimizer": (select_from_list(
            "Please choose optimizer",
            ["rmsprop", "adam"]),
            float(input("Learning rate: ")))
    }
    model_config["max_words"] = (15, 95) if int(model_config["lstm_count"]) == 2 else (100, 0)
    
    if model_config["lstm_count"] == 2:
        model_config["lstm_recurrent_dropout_1"] = float(input("Summary LSTM recurrent dropout: "))
        model_config["lstm_dropout_1"] = float(input("Summary LSTM dropout: "))
        model_config["lstm_recurrent_dropout_2"] = float(input("Description LSTM recurrent dropout: "))
        model_config["lstm_dropout_2"] = float(input("Description LSTM dropout: "))
    else:
        model_config["lstm_recurrent_dropout"] = float(input("LSTM recurrent dropout: "))
        model_config["lstm_dropout"] = float(input("LSTM dropout: "))

    if model_config["lstm_count"] == 2:
        model_config["bi_lstm_merge_mode"] = select_from_list(
            "Please select bidirectional LSTM merge mode",
            ["sum", "mul", "concat", "ave"])

    params = {
        "word_embeddings": embedding_config,
        "model_params": model_config,
        "training_dataset_id" : dataset,
        "run_id": "manual",
        "min_timespent_minutes": 10,
        "max_timespent_minutes": 960,
        "bin_count": 0,
        "min_delta": float(input("Minimum delta: ")),
        "patience": int(input("Patience: "))
    }

    print("\nSELECTED CONFIGURATION:")
    print(params)
    
    train_on_dataset(params, generate_graphs=True)
