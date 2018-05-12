from keras import losses
from keras import optimizers
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from preprocess import projects
from prima_model import load_data as load
from prima_model import model as mdl
from prima_model import save_results as save
from prima_model.callback import PrimaCallback
from utilities import load_data
from utilities.constants import *

# training parameters
batch_size = 25
learning_rate = 0.01
epochs = 1000
split_percentage = 75

def train_on_dataset(dataset, embeddings):

    # load and arrange data
    x_train, y_train, x_test, y_test = load.load_and_arrange(dataset, split_percentage, embeddings)

    max_y = max([np.max(y_train), np.max(y_test)])
    min_y = min([np.min(y_train), np.min(y_test)])
    norm_params = (min_y, max_y - min_y)
    y_train = (y_train - norm_params[0]) / norm_params[1]
    y_test = (y_test - norm_params[0]) / norm_params[1]
    

    # weight initialization with median value
    #y_train = np.full(y_train.shape, train_median)

    # create model
    max_text_length = x_test.shape[1]
    embedding_size = x_test.shape[2]
    model = mdl.create_model(max_text_length, embedding_size)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(loss=losses.mean_squared_error, optimizer=rmsprop)
    print(model.summary())

    #preload weights
    #init_model = load_model("weights/median-init-weights.hdf5")
    #model.set_weights(init_model.get_weights())

    # create results files
    load_data.create_folder_if_needed(WEIGTHS_FOLDER)
    training_session_name = load_data.get_next_dataset_name(WEIGTHS_FOLDER)
    weigths_directory_name = get_weigths_folder_name(dataset, training_session_name)
    load_data.create_folder_if_needed(weigths_directory_name)

    # save configuration
    '''configuration_filename = get_configuration_filename(dataset, training_session_name)
    filtered_data = load_data.load_json(get_dataset_filename(dataset, LABELED_FILENAME, FILTERED_POSTFIX, JSON_FILE_EXTENSION))
    min_text_length_datapoint = min(filtered_data, key=lambda datapoint: len(datapoint.get(SUMMARY_FIELD_KEY, "").split()) + len(datapoint.get(DESCRIPTION_FIELD_KEY, "").split()))
    min_text_length = len(min_text_length_datapoint.get(SUMMARY_FIELD_KEY, "").split()) + len(min_text_length_datapoint.get(DESCRIPTION_FIELD_KEY, "").split())
    project_issue_counts = projects.get_issue_counts(filtered_data)
    min_timespent = min([min(y_train), min(y_test)])
    max_timespent = max([max(y_train), max(y_test)])

    configuration = {
        "training" : {
            "split percentage" : split_percentage,
            "learning rate" : learning_rate,
            "batch size" : batch_size
        },
        "data" : {
            "mean_baseline" : mean_baseline,
            "median_baseline" : median_baseline,
            "datapoint_count" : {
                "training" : len(x_train),
                "testing" : len(x_test)
            },
            "text_length" : {
                "min_word_count" : min_text_length,
                "max_word_count" : max_text_length
            },
            "projects" : {
                "count" : len(project_issue_counts),
                "min_size" : min(project_issue_counts, key=lambda a: a[1])[1],
                "max_size" : max(project_issue_counts, key=lambda a: a[1])[1]
            },
            "timespent" : {
                "min" : min_timespent,
                "max" : max_timespent,
                "distribution" : projects.get_bins_and_volumes(filtered_data, 10, max_timespent - min_timespent)[1]
            } 
        }
    }
    with open(configuration_filename, "w") as configuration_file:
        json.dump(configuration, configuration_file, indent=JSON_INDENT)'''
    
    plot_filename = get_results_plot_filename(dataset, training_session_name)
    weigths_filename = get_weigths_filename(dataset, training_session_name)
    #save_weights = ModelCheckpoint(weigths_filename)
    results_filename = get_results_filename(dataset, training_session_name)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(results_filename, epoch, logs))

    # Save the model
    model.save(weigths_directory_name + "/model.h5")
    
    # train and validate
    callbacks = [save_results, PrimaCallback(model, x_train, x_test, y_train, y_test, plot_filename, norm_params)]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Save the model
    model.save(weigths_directory_name + "/model.h5")
    

train_on_dataset(sys.argv[1], sys.argv[2])