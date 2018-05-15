from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import sys

import prima_model.graph_helpers as gph
import prima_model.load_data as load
from utilities.constants import *

SPLIT_PERCENTAGE = 75

def predict(model, x_train, x_test, y_train, y_test, norm_params, ax_left, ax_right):

    ax_left.clear()
    ax_right.clear()
    
    # predict
    training_predictions = model.predict([x_train])
    testing_predictions = model.predict([x_test])

    training_predictions = norm_params[0] + norm_params[1] * training_predictions
    testing_predictions = norm_params[0] + norm_params[1] * testing_predictions
    y_train = norm_params[0] + norm_params[1] * y_train
    y_test = norm_params[0] + norm_params[1] * y_test

    # calculate deviations
    training_deviations = [abs(prediction[0] - y_train[i]) for i, prediction in enumerate(training_predictions)]
    testing_deviations = [abs(prediction[0] - y_test[i]) for i, prediction in enumerate(testing_predictions)]

    gph.create_prediction_scatter(ax_left, "Training dataset predictions", norm_params[0] + norm_params[1])
    ax_left.scatter(y_train, training_predictions, c=training_deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)

    gph.create_prediction_scatter(ax_right, "Testing dataset predictions", norm_params[0] + norm_params[1])
    ax_right.scatter(y_test, testing_predictions, c=testing_deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)

def predict_timespent(dataset, training_session_name, epoch, val_loss):

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
    plt.savefig(plot_filename, bbox_inches=PLOT_BBOX_INCHES)
    

#predict_timespent(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))