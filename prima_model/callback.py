from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from prima_model import calculate_baselines as bsl
from prima_model import graph_helpers as gph
from prima_model.predict import predict
from utilities.constants import *

class PrimaCallback(Callback):

    def __init__(self, model, x_train, x_test, y_train, y_test, filename):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.filename = filename
        self.max_hours = max([np.max(y_train), np.max(y_test)]) / SECONDS_IN_HOUR

        plt.figure(figsize=(12, 12))
        self.ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        self.ax2 = plt.subplot2grid((2, 2), (1, 0))
        self.ax3 = plt.subplot2grid((2, 2), (1, 1))

        self.training_losses = []
        self.testing_losses = []

        # calculate baseline losses
        train_mean, train_median = bsl.mean_and_median(y_train)
        self.mean_baseline = bsl.mean_squared_error(y_test, train_mean)
        self.median_baseline = bsl.mean_squared_error(y_test, train_median)

    def on_epoch_end(self, epoch, logs={}):

        self.training_losses.append(logs["loss"])
        self.testing_losses.append(logs["val_loss"])
        gph.plot_losses(self.ax1, self.training_losses, self.testing_losses, self.mean_baseline, self.median_baseline, SECONDS_IN_HOUR)
        predict(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.max_hours, self.ax2, self.ax3)
        plt.savefig(self.filename, bbox_inches=PLOT_BBOX_INCHES)


