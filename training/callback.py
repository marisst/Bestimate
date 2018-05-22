import matplotlib
matplotlib.use('Agg')

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from training import graph_helpers as gph
from training.predict import predict
from utilities.constants import *


class PrimaCallback(Callback):

    def __init__(self, model, x_train, x_test, y_train, y_test, filename, mean_baseline, median_baseline, loss):
        self.model = model
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self.filename = filename
        self.loss = loss

        plt.figure(figsize=(12, 12))
        self.ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        self.ax2 = plt.subplot2grid((2, 2), (1, 0))
        self.ax3 = plt.subplot2grid((2, 2), (1, 1))

        self.training_losses = []
        self.testing_losses = []

        self.mean_baseline = mean_baseline
        self.median_baseline = median_baseline


    def on_epoch_end(self, epoch, logs={}):

        self.training_losses.append(logs["loss"])
        self.testing_losses.append(logs["val_loss"])


    def on_train_end(self, logs={}):
        gph.plot_losses(self.ax1, self.training_losses, self.testing_losses, self.mean_baseline, self.median_baseline, self.loss)
        predict(self.model, self.x_train, self.x_test, self.y_train, self.y_test, self.ax2, self.ax3)
        plt.savefig(self.filename, bbox_inches=PLOT_BBOX_INCHES)