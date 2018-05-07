import numpy as np
import keras

# https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly.html

class DataGenerator(keras.utils.Sequence):

    def __init__(self, list_IDs, sentences, prepoint_params, window_size, batch_size=32, n_classes=10, shuffle=True):
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.n_classes = n_classes
        self.sentences = sentences
        self.prepoint_params = prepoint_params
        self.shuffle = shuffle
        self.window_size = window_size
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        
        X = np.empty((self.batch_size, self.window_size))
        y = np.empty((self.batch_size), dtype=int)

        for i, ID in enumerate(list_IDs_temp):

            sentence_index, start_index = self.prepoint_params[ID]
            X[i] = self.sentences[sentence_index][start_index:start_index+self.window_size]
            y[i] = self.sentences[sentence_index][start_index+self.window_size]

        return X, keras.utils.to_categorical(y, num_classes=self.n_classes)