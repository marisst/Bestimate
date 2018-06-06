import numpy as np
import keras
import gc
from utilities.constants import TEXT_FIELD_KEY, TIMESPENT_FIELD_KEY


class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, labels, batch_size, split_fields, max_words, vector_dictionary, shuffle=True):

        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.split_fields = split_fields
        self.max_words = max_words
        self.shuffle = shuffle
        self.vector_dictionary = vector_dictionary
        self.on_epoch_end()


    def __len__(self):
        
        return int(np.floor(len(self.data[0]) / self.batch_size))


    def __getitem__(self, index):

        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        if self.split_fields == True:
            batch_x = []
            for i, field_batch in enumerate(self.data):
                batch_data = [field_batch[k] for k in indexes]
                batch_x.append(self.__data_generation(batch_data, self.max_words[i]))
        else:
            batch_data = [self.data[0][k] for k in indexes]
            batch_x = self.__data_generation(batch_data, self.max_words[0])
        
        batch_y = [self.labels[k] for k in indexes]
        gc.collect()

        return batch_x, batch_y


    def on_epoch_end(self):

        gc.collect()
        self.indexes = np.arange(len(self.data[0]))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)


    def __data_generation(self, batch_data, max_words):

        x = np.zeros((len(batch_data), max_words, self.vector_dictionary.shape[1]))
        for i, datapoint in enumerate(batch_data):
            for j, encrypted_word in enumerate(datapoint):
                x[i, j] = self.vector_dictionary[encrypted_word]
        return x

        