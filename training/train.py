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

from training import calculate_baselines as bsl
from training import load_data as load
from training import model as mdl
from training import save_results as save
from training.callback import PrimaCallback
from training.data_generator import DataGenerator
from utilities.constants import *

# training parameters
learning_rate = 0.01
epochs = 1000
split_percentages = 60, 20
MIN_DELTA = 0.05
PATIENCE = 5


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


def calculate_validation_result(model, x_valid, y_valid, loss_function, model_params, vector_dictionary):

    validation_generator = DataGenerator(x_valid, y_valid, model_params["batch_size"], model_params["max_words"], vector_dictionary)
    validation_loss = model.evaluate_generator(generator=validation_generator, use_multiprocessing=True, workers=model_params["workers"])
    mean_baseline = loss_function(y_valid, np.mean(y_valid))
    median_baseline = loss_function(y_valid, np.median(y_valid))

    return validation_loss / min([mean_baseline, median_baseline])


def train_on_dataset(dataset, embedding_type, params, notes_filename = None, session_id = None, run_id = None, labeled_data=None, gensim_model = None):

    config = K.tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.1
    K.set_session(K.tf.Session(config=config))

    model_params = params["model_params"]

    if embedding_type == "spacy":
        nlp = spacy.load('en_vectors_web_lg')
        lookup = partial(spacy_lookup, nlp)

    if embedding_type == "gensim":
        if gensim_model is None:
            model_filename = get_dataset_filename(dataset, ALL_FILENAME, GENSIM_MODEL, PICKLE_FILE_EXTENSION)
            gensim_model = Word2Vec.load(model_filename)
        lookup = partial(gensim_lookup, gensim_model.wv)
        del gensim_model

    # load and arrange data
    data, vector_dictionary = load.load_and_arrange(
        dataset,
        split_percentages,
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
            print("Human loss:", human_loss, file=notes_file)

    if model_params["loss"] == "mean_absolute_percentage_error":
        loss_function = bsl.mean_absolute_percentage_error

    # calculate baseline losses
    mean_baseline = loss_function(y_test, np.mean(y_test))
    median_baseline = loss_function(y_test, np.median(y_test))
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
    weigths_directory_name = "%s/%s/%s" % (RESULTS_FOLDER, session_id, run_id)
   
    plot_filename = "%s/%s%s" % (weigths_directory_name, RESULTS_FILENAME, PNG_FILE_XTENSION)
    #weigths_filename = get_weigths_filename(dataset, training_session_name)
    #save_weights = ModelCheckpoint(weigths_filename)
    results_filename = "%s/%s%s" % (weigths_directory_name, RESULTS_FILENAME, TEXT_FILE_EXTENSION)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(results_filename, epoch, logs))

    best_model_filename = weigths_directory_name + "/model.h5"
    save_best_model = ModelCheckpoint(best_model_filename, save_best_only=True)

    training_generator = DataGenerator(x_train, y_train, model_params["batch_size"], model_params["max_words"], vector_dictionary)
    test_generator = DataGenerator(x_test, y_test, model_params["batch_size"], model_params["max_words"], vector_dictionary)

    # train and validate
    custom_callback = PrimaCallback(model, x_train, x_test, y_train, y_test, plot_filename, mean_baseline, median_baseline, model_params["loss"])
    callbacks = [save_results, save_best_model, EarlyStopping(min_delta=MIN_DELTA, patience=PATIENCE), custom_callback]
    
    history = model.fit_generator(
        generator = training_generator,
        validation_data = test_generator,
        use_multiprocessing=True,
        workers=model_params["workers"],
        callbacks=callbacks,
        epochs=epochs)

    del model
    K.clear_session()

    result = min(history.history["val_loss"]) / min([mean_baseline, median_baseline])
    with open(notes_filename, "a") as notes_file:
        print("Result:", result, file=notes_file)

    best_model = load_model(best_model_filename)
    val_result = calculate_validation_result(best_model, x_valid, y_valid, loss_function, model_params, vector_dictionary)

    return result, val_result