from keras.callbacks import Callback

SAVE_WEIGHTS_BATCHES = 1000

class PretrainingCallback(Callback):

    def __init__(self, model, weights_filename, results_filename):
        self.model = model
        self.weights_filename = weights_filename
        self.results_filename = results_filename
        self.epoch = 0
        self.batch = 0

    def on_batch_end(self, batch, logs={}):
        self.batch = batch
        self.save_results(logs["acc"])
        if self.batch % SAVE_WEIGHTS_BATCHES == 0:
            self.save_weights(logs["acc"])
        

    def on_epoch_end(self, epoch, logs={}):
        self.epoch = epoch
        self.save_results(logs["acc"])
        self.save_weights(logs["acc"])

    def save_weights(self, acc):
        filename = self.weights_filename.format(epoch=self.epoch, batch=self.batch, acc=acc)
        self.model.save_weights(filename)

    def save_results(self, acc):
        with open(self.results_filename, "a", newline="", encoding="utf-8-sig") as resultFile:
            print(",".join([str(self.epoch), str(self.batch), "%.2f" % acc]), file=resultFile)