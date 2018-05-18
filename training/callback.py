from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from training import calculate_baselines as bsl
from training import graph_helpers as gph
from training.predict import predict
from utilities.constants import *

class PrimaCallback(Callback):

    def __init__(self, model, x_train, x_test, y_train, y_test, filename, norm_params):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.filename = filename

        plt.figure(figsize=(12, 12))
        self.ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        self.ax2 = plt.subplot2grid((2, 2), (1, 0))
        self.ax3 = plt.subplot2grid((2, 2), (1, 1))

        self.training_losses = []
        self.testing_losses = []

        self.norm_params = norm_params

        # calculate baseline losses
        train_mean, train_median = bsl.mean_and_median(y_train)
        self.mean_baseline = norm_params[1] * bsl.mean_squared_error(y_test, train_mean)
        self.median_baseline = norm_params[1] * bsl.mean_squared_error(y_test, train_median)

    def on_epoch_end(self, epoch, logs={}):

        self.training_losses.append(self.norm_params[1] * logs["loss"])
        self.testing_losses.append(self.norm_params[1] * logs["val_loss"])
        gph.plot_losses(self.ax1, self.training_losses, self.testing_losses, self.mean_baseline, self.median_baseline)
        predict(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.norm_params, self.ax2, self.ax3)
        plt.savefig(self.filename, bbox_inches=PLOT_BBOX_INCHES)


