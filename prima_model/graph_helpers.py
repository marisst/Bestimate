import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 10
GRAPH_SPACE = 0.1

def plot_losses(axs, training_losses, testing_losses, mean_baseline, median_baseline, divide_by = 1):

    min_value = min(training_losses) #min([min(training_losses), min(testing_losses), mean_baseline, median_baseline]) / divide_by
    max_value = max(training_losses) #max([max(training_losses), max(testing_losses), mean_baseline, median_baseline]) / divide_by
    loss_range = max_value - min_value

    axs.clear()
    axs.set_title("Predict next word in JIRA issue text") #axs.set_title("Training and testing losses")
    axs.set_ylabel("Accuracy", fontsize=FONTSIZE) #axs.set_ylabel("Mean absolute error, hours", fontsize=FONTSIZE)
    axs.set_xlabel("Batch") #axs.set_xlabel("Epoch")

    epochs = len(training_losses)
    y_bottom = min_value - loss_range * GRAPH_SPACE
    y_top = max_value + loss_range * GRAPH_SPACE

    axs.axis([1, epochs, y_bottom, y_top])
    axs.grid(color='lightgray')

    #horizontal_padding = epochs * 0.003
    #vertical_padding = 0.003
    #axs.axhline(y=mean_baseline / divide_by, color='grey', linestyle='dotted')
    #axs.axhline(y=median_baseline / divide_by, color='grey', linestyle='dotted')
    #axs.text(epochs - horizontal_padding, (mean_baseline / divide_by) + vertical_padding, 'Mean prediction loss', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=FONTSIZE)
    #axs.text(epochs - horizontal_padding, (median_baseline / divide_by) + vertical_padding, 'Median prediction loss', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=FONTSIZE)

    adjusted_training_losses = [loss / divide_by for loss in training_losses]
    #adjusted_testing_losses = [loss / divide_by for loss in testing_losses]

    # draw loss lines
    axs.plot(range(1, epochs + 1), adjusted_training_losses, 'b-')
    #axs.plot(range(1, epochs + 1), adjusted_testing_losses, 'g-')

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
