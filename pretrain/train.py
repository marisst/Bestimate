from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys

from pretrain.load_data import get_sentences, get_vocabulary_size, get_prepoint_params
from pretrain.model import create_model
from pretrain.data_generator import DataGenerator
from utilities import load_data
from utilities.constants import *

# training parameters
embedding_size = 10
window_size = 5
lstm_nodes = 50
split_percentage = 90


def train_on_dataset(dataset):

    # create results files
    load_data.create_folder_if_needed(WEIGTHS_FOLDER)
    training_session_name = "%s_%s" % (load_data.get_next_dataset_name(WEIGTHS_FOLDER), PRELEARNING)
    weigths_directory_name = get_weigths_folder_name(dataset, training_session_name)
    load_data.create_folder_if_needed(weigths_directory_name)

    sentences = get_sentences(dataset)
    vocabuary_size = get_vocabulary_size(dataset)
    prepoint_params = get_prepoint_params(sentences, window_size)

    # fix random seed for reproducibility
    np.random.seed(7)
    permutation = np.random.permutation(len(prepoint_params))
    split_index = len(prepoint_params) * split_percentage // 100
    
    training_ids = permutation[:split_index]
    validation_ids = permutation[split_index:]

    training_generator = DataGenerator(training_ids, sentences, prepoint_params, window_size, n_classes=vocabuary_size)
    validation_generator = DataGenerator(validation_ids, sentences, prepoint_params, window_size, n_classes=vocabuary_size)

    model = create_model(window_size, embedding_size, vocabuary_size, lstm_nodes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    weigths_filename = get_weigths_folder_name(dataset, training_session_name) + "/weights-{epoch:04d}-{val_acc:.2f}" + HDF5_FILE_EXTENSION
    save_weigths = ModelCheckpoint(weigths_filename)
    
    callbacks = [save_weigths]

    model.fit_generator(generator=training_generator, validation_data=validation_generator, epochs=500, callbacks=callbacks)

train_on_dataset(sys.argv[1])