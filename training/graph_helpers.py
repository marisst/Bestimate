import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import math

from utilities.constants import PLOT_BBOX_INCHES
from training.data_generator import DataGenerator

FONTSIZE = 10
GRAPH_SPACE = 0.1

def plot_losses(training_losses, testing_losses, training_baseline, testing_baseline, loss, filename):

    plt.figure(figsize=(12, 6))
    axs = plt.gca()
    axs.clear()

    if loss == "mean_squared_error":
        training_losses = np.sqrt(training_losses)
        testing_losses = np.sqrt(testing_losses)
        training_baseline = math.sqrt(training_baseline)
        testing_baseline = math.sqrt(testing_baseline)
        loss_name = "Root-mean-square error, hours"

    if loss == "mean_absolute_error":
        loss_name = "Mean absolute error, hours"

    if loss == "mean_absolute_percentage_error":
        loss_name = "Mean absolute percentage error, hours"

    min_value = min([min(training_losses), min(testing_losses), training_baseline, testing_baseline])
    max_value = max([max(training_losses), max(testing_losses), training_baseline, testing_baseline])
    loss_range = max_value - min_value

    axs.clear()
    axs.set_title("Training and testing losses")
    axs.set_ylabel(loss_name, fontsize=FONTSIZE)
    axs.set_xlabel("Epoch")

    epochs = len(training_losses)
    y_bottom = min_value - loss_range * GRAPH_SPACE
    y_top = max_value + loss_range * GRAPH_SPACE

    axs.axis([1, epochs, y_bottom, y_top])
    axs.grid(color='lightgray')

    horizontal_padding = epochs * 0.003
    vertical_padding = 0.003
    axs.axhline(y=training_baseline, color='grey', linestyle='dotted')
    axs.axhline(y=testing_baseline, color='grey', linestyle='dotted')
    axs.text(epochs - horizontal_padding, (training_baseline) + vertical_padding, 'Training baseline', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=FONTSIZE)
    axs.text(epochs - horizontal_padding, (testing_baseline) + vertical_padding, 'Testing baseline', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=FONTSIZE)

    # draw loss lines
    axs.plot(range(1, epochs + 1), training_losses, 'b-')
    axs.plot(range(1, epochs + 1), testing_losses, 'g-')

    plt.savefig(filename, bbox_inches=PLOT_BBOX_INCHES)

def create_prediction_scatter(model, x, y, filename, title, model_params, vector_dictionary):

    plt.figure(figsize=(6, 6))
    axs = plt.gca()
    axs.clear()
    axs.set_title(title)

    data_generator = DataGenerator(
        x,
        y,
        len(y),
        True if model_params["lstm_count"] == 2 else False,
        model_params["max_words"],
        vector_dictionary,
        shuffle=False)
    predictions = model.predict_generator(data_generator, use_multiprocessing=True, workers=model_params["workers"])
    deviations = np.array([abs(prediction[0] - y[i]) for i, prediction in enumerate(predictions)])
    max_plot_hours = max(y)

    axs.axis([0, max_plot_hours, 0, max_plot_hours])
    axs.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
    axs.set_ylabel("Model estimate, hours", fontsize=FONTSIZE)
    axs.set_xlabel("Time spent, hours", fontsize=FONTSIZE)
    axs.scatter(y, predictions, c=deviations, cmap='coolwarm_r', marker='x', alpha = 0.5)

    plt.savefig(filename, bbox_inches=PLOT_BBOX_INCHES)

def update_graph(axs, model, x):
    
    pred = model.predict(x)
    dev = []
    for i, p in enumerate(pred):
        dev.append(abs(p[0] - x[i]))
    axs.clear()
    axs.scatter(x, pred, c=dev, cmap='coolwarm_r')
