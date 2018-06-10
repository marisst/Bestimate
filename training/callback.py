import matplotlib
matplotlib.use('Agg')

from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np

from training import graph_helpers as gph
from training.predict import predict
from utilities.constants import *


class CustomCallback(Callback):

    def __init__(self, train_generator, test_generator, valid_generator, filename, mean_baseline, median_baseline, loss, workers):
        
        self.train_generator = train_generator
        self.test_generator = test_generator
        self.valid_generator = valid_generator
        self.filename = filename
        self.loss = loss
        self.workers = workers

        self.training_losses = []
        self.testing_losses = []

        self.mean_baseline = mean_baseline
        self.median_baseline = median_baseline

    def on_epoch_end(self, epoch, logs={}):

        self.training_losses.append(logs["loss"])
        self.testing_losses.append(logs["val_loss"])


    def generate_graph(self, best_model):

        plt.figure(figsize=(12, 12))
        ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
        ax2 = plt.subplot2grid((2, 2), (1, 0))
        ax3 = plt.subplot2grid((2, 2), (1, 1))

        gph.plot_losses(ax1, self.training_losses, self.testing_losses, self.mean_baseline, self.median_baseline, self.loss)
        predict(best_model, ax2, ax3, self.train_generator, self.test_generator, self.valid_generator, self.max_hours, self.workers)
        plt.savefig(self.filename, bbox_inches=PLOT_BBOX_INCHES)