from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import sys

import prima_model.graph_helpers as gph
import prima_model.load_data as load
from utilities.constants import *

SPLIT_PERCENTAGE = 75

def predict_timespent(dataset, training_session_name, epoch, val_loss):

    # load data
    x_train, y_train, x_test, y_test = load.load_and_arrange(dataset, SPLIT_PERCENTAGE)
    y_train = y_train / SECONDS_IN_HOUR
    y_test = y_test / SECONDS_IN_HOUR

    # load model
    weights_filename = get_weigths_filename(dataset, training_session_name).format(epoch=epoch, val_loss=val_loss)
    model = load_model(weights_filename)
    print("Model loaded")

    # predict
    training_predictions = model.predict([x_train]) / SECONDS_IN_HOUR
    testing_predictions = model.predict([x_test]) / SECONDS_IN_HOUR
    print("PRedictions calculated")

    # calculate deviations
    training_deviations = [abs(prediction[0] - y_train[i]) for i, prediction in enumerate(training_predictions)]
    testing_deviations = [abs(prediction[0] - y_test[i]) for i, prediction in enumerate(testing_predictions)]

    max_plot_hours = int(input("Please select maximum number of hours to display in the plot: "))

    # draw plot
    plt.figure(figsize=(15, 8))
    ax1 = plt.subplot2grid((1, 2), (0, 0))
    ax2 = plt.subplot2grid((1, 2), (0, 1))

    gph.create_prediction_scatter(ax1, "Training dataset predictions", max_plot_hours)
    ax1.scatter(y_train, training_predictions, c=training_deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)

    gph.create_prediction_scatter(ax2, "Testing dataset predictions", max_plot_hours)
    ax2.scatter(y_test, testing_predictions, c=testing_deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)
    print("Plot created")

    # save plot
    plot_filename = get_prediction_plot_filename(dataset, training_session_name).format(epoch=epoch, val_loss=val_loss)
    plt.savefig(plot_filename, bbox_inches=PLOT_BBOX_INCHES)
    print("Plot saved at %s" % plot_filename)

predict_timespent(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))