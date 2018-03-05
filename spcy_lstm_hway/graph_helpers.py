import numpy as np
import matplotlib.pyplot as plt

fontsize = 10

def create_losses_plot(axs, title, epochs, y_range, mean_baseline, median_baseline):
    
    axs.set_title("Training and testing losses")
    axs.set_ylabel("Mean absolute error, hours", fontsize=fontsize)
    axs.set_xlabel("Epoch")
    axs.axis([0, epochs, y_range[0], y_range[1]])
    axs.grid(color='lightgray')

    horizontal_padding = epochs * 0.003
    vertical_padding = 0.003
    axs.axhline(y=mean_baseline / 3600, color='grey', linestyle='dotted')
    axs.axhline(y=median_baseline / 3600, color='grey', linestyle='dotted')
    axs.text(epochs - horizontal_padding, (mean_baseline / 3600) + vertical_padding, 'Mean prediction loss', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=fontsize)
    axs.text(epochs - horizontal_padding, (median_baseline / 3600) + vertical_padding, 'Median prediction loss', verticalalignment='bottom', horizontalalignment='right', color='grey', fontsize=fontsize)

def create_prediction_scatter(axs, title, max_plot_hours):
    axs.set_title(title)
    axs.axis([0, max_plot_hours, 0, max_plot_hours])
    axs.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
    axs.set_ylabel("Model estimate, hours", fontsize=fontsize)
    axs.set_xlabel("Time spent, hours", fontsize=fontsize)

def update_line(hl, epoch, loss):
    hl.set_xdata(np.append(hl.get_xdata(), [epoch]))
    hl.set_ydata(np.append(hl.get_ydata(), [(loss / 3600)]))
    plt.draw()
    plt.pause(0.1)

def update_graph(axs, model, x):
    pred = model.predict(x)
    dev = []
    for i, p in enumerate(pred):
        dev.append(abs(p[0] - x[i]))
    axs.clear()
    axs.scatter(x, pred, c=dev, cmap='coolwarm_r')
    plt.pause(0.1)
    plt.draw()
