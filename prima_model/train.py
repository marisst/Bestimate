from keras import losses
from keras import optimizers
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import load_model
import json
import matplotlib.pyplot as plt
import numpy as np
import sys

from prima_model import arrange_data as arr
from prima_model import calculate_baselines as bsl
from prima_model import graph_helpers as gph
from prima_model import model as mdl
from prima_model import save_results as save
from utilities import load_data
from utilities.constants import *

# training parameters
split_percentage = 75
learning_rate = 0.02
epochs = 1000
batch_size = 50

def train_on_dataset(dataset):

    # load and arrange data
    filename = get_vectorized_dataset_filename(dataset)
    data = load_data.load_pickle(filename)
    shuffled_data = arr.shuffle(data)
    splitted_data = arr.split_train_test(shuffled_data, split_percentage)
    x_train, y_train, x_test, y_test = splitted_data

    # calculate baseline losses
    train_mean, train_median = bsl.mean_and_median(y_train)
    mean_baseline = bsl.mean_absolute_error(y_test, train_mean)
    median_baseline = bsl.mean_absolute_error(y_test, train_median)

    # weight initialization with median value
    #y_train = np.full(y_train.shape, train_median)

    # create model
    max_text_length = x_test.shape[1]
    model = mdl.create_model(max_text_length)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(loss=losses.mean_absolute_error, optimizer=rmsprop)
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
    configuration_filename = get_configuration_filename(dataset, training_session_name)
    configuration = {
        "training" : {
            "split percentage" : split_percentage,
            "learning rate" : learning_rate,
            "batch size" : batch_size
        },
        "data" : {
            "mean_baseline" : mean_baseline,
            "median_baseline" : median_baseline,
            "max_text_length" : max_text_length,
            "datapoint_count" : {
                "training" : len(x_train),
                "testing" : len(x_test)
            }
        }
    }
    with open(configuration_filename, "w") as configuration_file:
        json.dump(configuration, configuration_file, indent=JSON_INDENT)
    
    # update losses plot after every update
    training_losses = []
    testing_losses = []
    axs = plt.gca()
    log_training_loss = LambdaCallback(on_epoch_end=lambda epoch, logs: training_losses.append(logs['loss']))
    log_testing_loss = LambdaCallback(on_epoch_end=lambda epoch, logs: testing_losses.append(logs['val_loss']))
    update_graph = LambdaCallback(on_epoch_end=lambda epoch, logs: gph.plot_losses(axs, training_losses, testing_losses, mean_baseline, median_baseline, SECONDS_IN_HOUR))
    plot_filename = get_results_plot_filename(dataset, training_session_name)
    save_graph = LambdaCallback(on_epoch_end=lambda epoch, logs: plt.savefig(plot_filename, bbox_inches=PLOT_BBOX_INCHES))

    # save weights and results after every epoch
    weigths_filename = get_weigths_filename(dataset, training_session_name)
    save_weights = ModelCheckpoint(weigths_filename)
    results_filename = get_results_filename(dataset, training_session_name)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(results_filename, epoch, logs))

    # Save the model
    model.save(weigths_directory_name + "/model.h5")
    
    # train and validate
    callbacks = [save_weights, save_results, log_training_loss, log_testing_loss, update_graph, save_graph]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Save the model
    model.save(weigths_directory_name + "/model.h5")

    

train_on_dataset(sys.argv[1])