import matplotlib.pyplot as plt
import numpy as np

FONTSIZE = 10
GRAPH_SPACE = 0.1

def plot_accuracy(axs, accuracy):

    min_value = min(accuracy)
    max_value = max(accuracy)
    loss_range = max_value - min_value

    axs.clear()
    axs.set_title("Predict next word in JIRA issue text")
    axs.set_ylabel("Accuracy", fontsize=FONTSIZE)
    axs.set_xlabel("Batch")

    batches = len(accuracy)
    y_bottom = min_value - loss_range * GRAPH_SPACE
    y_top = max_value + loss_range * GRAPH_SPACE

    axs.axis([1, batches, y_bottom, y_top])
    axs.grid(color='lightgray')

    # draw loss lines
    axs.plot(range(1, batches + 1), accuracy, 'b-')