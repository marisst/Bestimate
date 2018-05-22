import matplotlib
matplotlib.use('Agg')

from keras.callbacks import ModelCheckpoint, EarlyStopping, LambdaCallback
from keras.losses import mean_squared_error, mean_absolute_error
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from training import calculate_baselines as bsl
from training import load_data as load
from training import model as mdl
from training import save_results as save
from training.callback import PrimaCallback
from utilities.constants import *

# training parameters
learning_rate = 0.01
epochs = 1000
split_percentages = 60, 20

def calculate_validation_result(model, x_valid, y_valid, loss_function):

    validation_loss = model.evaluate(x_valid, y_valid)
    mean_baseline = loss_function(y_valid, np.mean(y_valid))
    median_baseline = loss_function(y_valid, np.median(y_valid))

    return validation_loss / min([mean_baseline, median_baseline])

def train_on_dataset(dataset, embedding_type, params, notes_filename = None, session_id = None, run_id = None):

    model_params = params["model_params"]

    # load and arrange data
    x_train, y_train, x_test, y_test, x_valid, y_valid = load.load_and_arrange(dataset, split_percentages, embedding_type, model_params["max_words"])

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
    max_text_length = x_test.shape[1]
    embedding_size = x_test.shape[2]
    model = mdl.create_model(max_text_length, embedding_size, model_params)
    model.compile(loss=model_params["loss"], optimizer=model_params["optimizer"])

    # create results files
    weigths_directory_name = "%s/%s/%s" % (RESULTS_FOLDER, session_id, run_id)
   
    plot_filename = "%s/%s%s" % (weigths_directory_name, RESULTS_FILENAME, PNG_FILE_XTENSION)
    #weigths_filename = get_weigths_filename(dataset, training_session_name)
    #save_weights = ModelCheckpoint(weigths_filename)
    results_filename = "%s/%s%s" % (weigths_directory_name, RESULTS_FILENAME, TEXT_FILE_EXTENSION)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(results_filename, epoch, logs))

    best_model_filename = weigths_directory_name + "/model.h5"
    save_best_model = ModelCheckpoint(best_model_filename, save_best_only=True)

    # train and validate
    callbacks = [save_results, save_best_model, PrimaCallback(model, x_train, x_test, y_train, y_test, plot_filename, mean_baseline, median_baseline, model_params["loss"]), EarlyStopping(min_delta=0.001, patience=15)]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=model_params["batch_size"], callbacks=callbacks)

    result = min(history.history["val_loss"]) / min([mean_baseline, median_baseline])
    with open(notes_filename, "a") as notes_file:
        print("Result:", result, file=notes_file)

    best_model = load_model(best_model_filename)
    val_result = calculate_validation_result(best_model, x_valid, y_valid, loss_function)

    return result, val_result