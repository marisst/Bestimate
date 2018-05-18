from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
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
split_percentage = 75

def train_on_dataset(dataset, embedding_type, model_params, notes_filename = None, session_id = None, run_id = None):

    # load and arrange data
    x_train, y_train, x_test, y_test = load.load_and_arrange(dataset, split_percentage, embedding_type, model_params["max_words"])

    if model_params["loss"] == "mean_squared_error":
        loss_function = bsl.mean_squared_error

    if model_params["loss"] == "mean_absolute_error":
        loss_function = bsl.mean_absolute_error

    # calculate baseline losses
    mean_baseline = loss_function(y_test, np.mean(y_train))
    median_baseline = loss_function(y_test, np.median(y_train))

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

    # train and validate
    callbacks = [save_results, PrimaCallback(model, x_train, x_test, y_train, y_test, plot_filename, mean_baseline, median_baseline, model_params["loss"]), EarlyStopping(min_delta=0.001, patience=10)]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=model_params["batch_size"], callbacks=callbacks)

    result = min(history.history["val_loss"]) / min([mean_baseline, median_baseline])
    with open(notes_filename, "a") as notes_file:
        print("Result:", result, file=notes_file)

    model.save(weigths_directory_name + "/model.h5")

    return result