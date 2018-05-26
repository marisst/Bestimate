import numpy as np
np.set_printoptions(threshold=np.nan)
import keras
from utilities.constants import TEXT_FIELD_KEY, TIMESPENT_FIELD_KEY

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, labels, batch_size, max_words, vector_dictionary, shuffle=True):
        
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.max_words = max_words
        self.shuffle = shuffle
        self.vector_dictionary = vector_dictionary
        self.on_epoch_end()

    def __len__(self):

        return int(np.floor(len(self.data) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [self.data[k] for k in indexes]
        batch_x = self.__data_generation(batch_data)
        batch_y = [self.labels[k] for k in indexes]

        return batch_x, batch_y

    def on_epoch_end(self):

        self.indexes = np.arange(len(self.data))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):

        x = np.zeros((len(batch_data), self.max_words, self.vector_dictionary.shape[1]))
        for i, datapoint in enumerate(batch_data):
            for j, encrypted_word in enumerate(datapoint):
                x[i, j] = self.vector_dictionary[encrypted_word]

        return x

        