from keras.callbacks import Callback
import matplotlib.pyplot as plt
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
        self.loss_history = []
        self.axs = plt.gca()

    def on_batch_end(self, batch, logs={}):
        self.batch = batch
        self.save_results(logs["loss"], logs["acc"])
        self.loss_history.append(logs["loss"])
        
        if int(self.batch) % SAVE_WEIGHTS_BATCHES == 0:
            self.save_weights(logs["acc"])
            self.update_graph()
        

    def on_epoch_end(self, epoch, logs={}):
        self.save_results(logs["loss"], logs["acc"])
        self.save_weights(logs["acc"])
        self.epoch = epoch + 1

    def save_weights(self, acc):
        filename = self.weights_filename.format(epoch=self.epoch, batch=self.batch, acc=acc)
        self.model.save_weights(filename)

    def save_results(self, loss, acc):
        with open(self.results_filename, "a", newline="", encoding="utf-8-sig") as resultFile:
            print(",".join([str(self.epoch), str(self.batch), "%.4f" % loss, "%.2f" % acc]), file=resultFile)

    def update_graph(self):
        plot_losses(self.axs, self.loss_history, [], 0, 0, 1)
        plt.savefig(self.graph_filename)