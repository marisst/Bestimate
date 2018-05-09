from keras.callbacks import Callback
import matplotlib.pyplot as plt
import numpy as np
from prima_model.graph_helpers import plot_losses

SAVE_WEIGHTS_BATCHES = 1000

class PretrainingCallback(Callback):

    def __init__(self, model, weights_filename, results_filename, graph_filename):
        self.model = model
        self.weights_filename = weights_filename
        self.results_filename = results_filename
        self.graph_filename = graph_filename
        self.epoch = 0
        self.batch = 0

        #graph
        self.accuracy_history_long = []
        self.accuracy_history_short = []
        self.axs = plt.gca()

    def on_batch_end(self, batch, logs={}):
        self.batch = batch
        self.save_results(logs["loss"], logs["acc"])
        self.accuracy_history_short.append(logs["acc"])
        
        if int(self.batch) % SAVE_WEIGHTS_BATCHES == 0:

            average_step_accuracy = np.average(np.array(self.accuracy_history_short))
            if average_step_accuracy != 0:
                self.accuracy_history_long.append(average_step_accuracy * 100)
                
            self.accuracy_history_short = []

            self.save_weights(logs["acc"])
            self.update_graph()
        

    def on_epoch_end(self, epoch, logs={}):
        self.save_results(logs["loss"], logs["acc"])
        self.save_weights(logs["acc"])
        self.epoch = epoch + 1

    def save_weights(self, acc):
        filename = self.weights_filename.format(epoch=self.epoch, batch=self.batch // 1000, acc=acc)
        self.model.save(filename)

    def save_results(self, loss, acc):
        with open(self.results_filename, "a", newline="", encoding="utf-8-sig") as resultFile:
            print(",".join([str(self.epoch), str(self.batch), "%.4f" % loss, "%.2f" % acc]), file=resultFile)

    def update_graph(self):

        if (len(self.accuracy_history_long) < 2):
            return
            
        plot_losses(self.axs, self.accuracy_history_long, [], 0, 0, 1)
        plt.savefig(self.graph_filename)