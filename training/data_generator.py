import numpy as np
import keras
from utilities.constants import TEXT_FIELD_KEY, TIMESPENT_FIELD_KEY

class DataGenerator(keras.utils.Sequence):

    def __init__(self, data, labels, batch_size, lookup, max_words, embedding_size):
        
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.on_epoch_end()
        self.lookup = lookup
        self.max_words = max_words
        self.embedding_size = embedding_size

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
        np.random.shuffle(self.indexes)

    def __data_generation(self, batch_data):

        x = np.zeros((len(batch_data), self.max_words, self.embedding_size))
        for i, datapoint in enumerate(batch_data):
            words = datapoint.split()
            vectorized_words = []
            for word in words:
                vectorized_word = self.lookup(word)
                if vectorized_word is not None:
                    vectorized_words.append(vectorized_word)
                    if len(vectorized_words) > self.max_words:
                        break

            start_index = self.max_words - len(vectorized_words)
            for j, vectorized_word in enumerate(vectorized_words):
                x[i, start_index + j] = np.array(vectorized_word)

        return x

        