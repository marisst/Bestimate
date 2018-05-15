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
learning_rate = 0.01
epochs = 200
split_percentage = 75

def train_on_dataset(dataset, embedding_type, model_params):

    # load and arrange data
    x_train, y_train, x_test, y_test = load.load_and_arrange(dataset, split_percentage, embedding_type, model_params["max_words"])

    max_y = max([np.max(y_train), np.max(y_test)])
    min_y = min([np.min(y_train), np.min(y_test)])
    norm_params = (min_y, max_y - min_y)
    y_train = (y_train - norm_params[0]) / norm_params[1]
    y_test = (y_test - norm_params[0]) / norm_params[1]

    # create model
    max_text_length = x_test.shape[1]
    embedding_size = x_test.shape[2]
    model = mdl.create_model(max_text_length, embedding_size, model_params)
    model.compile(loss=model_params["loss"], optimizer=model_params["optimizer"])
    print(model.summary())

    # create results files
    load_data.create_folder_if_needed(WEIGTHS_FOLDER)
    training_session_name = load_data.get_next_dataset_name(WEIGTHS_FOLDER)
    weigths_directory_name = get_weigths_folder_name(dataset, training_session_name)
    load_data.create_folder_if_needed(weigths_directory_name)
    
    plot_filename = get_results_plot_filename(dataset, training_session_name)
    weigths_filename = get_weigths_filename(dataset, training_session_name)
    #save_weights = ModelCheckpoint(weigths_filename)
    results_filename = get_results_filename(dataset, training_session_name)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(results_filename, epoch, logs))

    # Save the model
    model.save(weigths_directory_name + "/model.h5")
    
    # train and validate
    callbacks = [save_results, PrimaCallback(model, x_train, x_test, y_train, y_test, plot_filename, norm_params)]
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=model_params["batch_size"], callbacks=callbacks)

    

    # Save the model
    model.save(weigths_directory_name + "/model.h5")


    

#train_on_dataset(sys.argv[1], sys.argv[2])