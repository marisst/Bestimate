from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys

from pretrain.load_data import load_and_arange
from pretrain.model import create_model
from utilities import load_data
from utilities.constants import *

# training parameters
embedding_size = 10
window_size = 5
lstm_nodes = 50

def get_vocabulary_size(dataset):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_data.load_json(dictionary_filename)
    return len(dictionary) + 1

def train_on_dataset(dataset):

    # create results files
    load_data.create_folder_if_needed(WEIGTHS_FOLDER)
    training_session_name = "%s_%s" % (load_data.get_next_dataset_name(WEIGTHS_FOLDER), PRELEARNING)
    weigths_directory_name = get_weigths_folder_name(dataset, training_session_name)
    load_data.create_folder_if_needed(weigths_directory_name)

    x, y = load_and_arange(dataset, window_size)

    vocabuary_size = max([x.max(), y.max()]) + 1
    y = to_categorical(y, num_classes=vocabuary_size)

    model = create_model(window_size, embedding_size, vocabuary_size, lstm_nodes)

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    weigths_filename = get_weigths_folder_name(dataset, training_session_name) + "/weights-{epoch:04d}-{val_acc:.2f}" + HDF5_FILE_EXTENSION
    save_weigths = ModelCheckpoint(weigths_filename)

    callbacks = [save_weigths]
    model.fit(x, y, validation_split=0.33, epochs=500, callbacks=callbacks)

train_on_dataset(sys.argv[1])