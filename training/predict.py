import matplotlib
matplotlib.use('Agg')

from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import sys

import training.graph_helpers as gph
import training.load_data as load
from training.data_generator import DataGenerator
from utilities.constants import *

SPLIT_PERCENTAGE = 75

def predict(model, ax_left, ax_right, train_generator, test_generator, valid_generator, max_hours, workers):

    ax_left.clear()
    ax_right.clear()
    
    # predict
    print("Predicting on training dataset")
    training_predictions = model.predict_generator(train_generator, use_multiprocessing=True, workers=workers)
    print("Predicting on testing dataset")
    testing_predictions = model.predict_generator(test_generator, use_multiprocessing=True, workers=workers)
    print("Predicting on validation dataset")
    validation_predictions = model.predict_generator(valid_generator, use_multiprocessing=True, workers=workers)

    # calculate deviations
    training_deviations = [abs(prediction[0] - y_train[i]) for i, prediction in enumerate(training_predictions)]
    testing_deviations = [abs(prediction[0] - y_test[i]) for i, prediction in enumerate(testing_predictions)]
    validation_deviations = []

    gph.create_prediction_scatter(ax_left, "Training dataset predictions", max_hours)
    ax_left.scatter(y_train, training_predictions, c=training_deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)

    gph.create_prediction_scatter(ax_right, "Testing dataset predictions", max_hours)
    ax_right.scatter(y_test, testing_predictions, c=testing_deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)

"""def predict_timespent(dataset, training_session_name, epoch, val_loss):

    # load data
    x_train, y_train, x_test, y_test = load.load_and_arrange(dataset, SPLIT_PERCENTAGE)
    max_plot_hours = int(input("Please select maximum number of hours to display in the plot: "))

    # load model
    weights_filename = get_weigths_filename(dataset, training_session_name).format(epoch=epoch, val_loss=val_loss)
    model = load_model(weights_filename)
    print("Model loaded")

    plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid((2, 2), (0, 0))
    ax2 = plt.subplot2grid((2, 2), (0, 1))

    plot_filename = get_prediction_plot_filename(dataset, training_session_name).format(epoch=epoch, val_loss=val_loss)
    predict(model, x_train, x_test, y_train, y_test, max_plot_hours, ax1, ax2)
    plt.savefig(plot_filename, bbox_inches=PLOT_BBOX_INCHES)"""
    

#predict_timespent(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))