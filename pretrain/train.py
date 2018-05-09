from keras.utils import to_categorical
from keras.callbacks import LambdaCallback, ModelCheckpoint
import numpy as np
import sys

from pretrain.load_data import get_sentences, get_vocabulary_size, get_prepoint_params
from pretrain.model import create_model
from pretrain.data_generator import DataGenerator
from pretrain.callback import PretrainingCallback
from utilities import load_data
from utilities.constants import *

# http://adventuresinmachinelearning.com/word2vec-keras-tutorial/

# training parameters
embedding_size = 32
window_size = 5
lstm_nodes = 64
batch_size = 32
epochs = 1

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
    training_generator = DataGenerator(permutation, sentences, prepoint_params, window_size, n_classes=vocabuary_size, batch_size=batch_size)

    model = create_model(window_size, embedding_size, vocabuary_size, lstm_nodes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    weights_filename = get_weigths_folder_name(dataset, training_session_name) + "/weights-{epoch:04d}-{batch:04d}-{acc:.2f}" + HDF5_FILE_EXTENSION
    results_filename = "%s/%s-%s/%s%s" % (WEIGTHS_FOLDER, dataset, training_session_name, RESULTS_FILENAME, CSV_FILE_EXTENSION)
    graph_filename = "%s/%s-%s/%s%s" % (WEIGTHS_FOLDER, dataset, training_session_name, RESULTS_FILENAME, PNG_FILE_XTENSION)
    model.fit_generator(generator=training_generator, epochs=epochs, callbacks=[PretrainingCallback(model, weights_filename, results_filename, graph_filename)])

train_on_dataset(sys.argv[1])