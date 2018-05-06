from keras.utils import to_categorical
import numpy as np
import sys

from pretrain.load_data import load_and_arange
from pretrain.model import create_model

# training parameters
embedding_size = 50
split_percentage = 75
window_size = 100

def train_on_dataset(dataset):

    x_train, y_train, x_test, y_test = load_and_arange(dataset, window_size, split_percentage)
    vocabuary_size = max([x_train.max(), y_train.max(), x_test.max(), y_test.max()]) + 1
    y_train = to_categorical(y_train, num_classes=vocabuary_size)
    y_test = to_categorical(y_test, num_classes=vocabuary_size)

    model = create_model(window_size, embedding_size, vocabuary_size)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=500, verbose=2)

train_on_dataset(sys.argv[1])