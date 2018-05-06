from keras.utils import to_categorical
import numpy as np
import sys

from pretrain.load_data import load_and_arange
from pretrain.model import create_model

# training parameters
embedding_size = 50
window_size = 100

def train_on_dataset(dataset):

    x, y = load_and_arange(dataset, window_size)
    vocabuary_size = max([x.max(), y.max()]) + 1
    y = to_categorical(y, num_classes=vocabuary_size)

    model = create_model(window_size, embedding_size, vocabuary_size)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(x, y, epochs=500, verbose=2)

train_on_dataset(sys.argv[1])