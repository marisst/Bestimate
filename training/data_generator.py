import numpy as np
np.set_printoptions(threshold=np.nan)
import keras
import gc
from utilities.constants import TEXT_FIELD_KEY, TIMESPENT_FIELD_KEY
np.set_printoptions(threshold=np.nan)

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

        return int(np.floor(len(self.data[0]) / self.batch_size))

    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        batch_data = [[self.data[0][k] for k in indexes], [self.data[1][k] for k in indexes]]
        batch_x = self.__data_generation(batch_data)
        batch_y = [self.labels[k] for k in indexes]
        gc.collect()

        return batch_x, batch_y


    def on_epoch_end(self):

        gc.collect()
        self.indexes = np.arange(len(self.data[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def encrypt(self, batch_data_column, max_words):

        encrypted_batch_data_column = np.zeros((len(batch_data_column), max_words, self.vector_dictionary.shape[1]))
        for i, datapoint in enumerate(batch_data_column):
            for j, encrypted_word in enumerate(datapoint):
                encrypted_batch_data_column[i, j] = self.vector_dictionary[encrypted_word]

        return encrypted_batch_data_column


    def __data_generation(self, batch_data):
        
        return [self.encrypt(batch_data[i], self.max_words[i]) for i in range(2)]

        