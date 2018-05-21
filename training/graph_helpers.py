import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import math

FONTSIZE = 10
GRAPH_SPACE = 0.1

def plot_losses(axs, training_losses, testing_losses, mean_baseline, median_baseline, loss):

    if loss == "mean_squared_error":
        training_losses = np.sqrt(training_losses)
        testing_losses = np.sqrt(testing_losses)
        mean_baseline = math.sqrt(mean_baseline)
        median_baseline = math.sqrt(median_baseline)
        loss_name = "Root-mean-square deviation, hours"

    if loss == "mean_absolute_error":
        loss_name = "Mean absolute error"

    

    min_value = min([min(training_losses), min(testing_losses), mean_baseline, median_baseline])
    max_value = max([max(training_losses), max(testing_losses), mean_baseline, median_baseline])
    loss_range = max_value - min_value

    axs.clear()
    axs.set_title("Training and testing losses")
    axs.set_ylabel(loss_name, fontsize=FONTSIZE)
    axs.set_xlabel("Epoch") #axs.set_xlabel("Batch") #

    epochs = len(training_losses)
    y_bottom = min_value - loss_range * GRAPH_SPACE
    y_top = max_value + loss_range * GRAPH_SPACE

    axs.axis([1, epochs, y_bottom, y_top])
    axs.grid(color='lightgray')

    horizontal_padding = epochs * 0.003
    vertical_padding = 0.003
    axs.axhline(y=mean_baseline, color='grey', linestyle='dotted')
    axs.axhline(y=median_baseline, color='grey', linestyle='dotted')
    axs.text(epochs - horizontal_padding, (mean_baseline) + vertical_padding, 'Mean prediction loss', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=FONTSIZE)
    axs.text(epochs - horizontal_padding, (median_baseline) + vertical_padding, 'Median prediction loss', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=FONTSIZE)

    # draw loss lines
    axs.plot(range(1, epochs + 1), training_losses, 'b-')
    axs.plot(range(1, epochs + 1), testing_losses, 'g-')

def create_prediction_scatter(axs, title, max_plot_hours):

    axs.set_title(title)
    axs.axis([0, max_plot_hours, 0, max_plot_hours])
    axs.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
    axs.set_ylabel("Model estimate, hours", fontsize=FONTSIZE)
    axs.set_xlabel("Time spent, hours", fontsize=FONTSIZE)

def update_graph(axs, model, x):
    
    pred = model.predict(x)
    dev = []
    for i, p in enumerate(pred):
        dev.append(abs(p[0] - x[i]))
    axs.clear()
    axs.scatter(x, pred, c=dev, cmap='coolwarm_r')
