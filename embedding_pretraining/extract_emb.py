from keras import backend as K
from keras.models import load_model
import math
import numpy as np
import sys

from utilities.constants import *
from utilities.load_data import load_json, save_json

def extract_emb(dataset, training_session, epoch, batch, accuracy):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)

    weights_filename = "%s/%s-%s_%s/weights-%04d-%04d-%.2f%s" % (WEIGTHS_FOLDER, dataset, training_session, PRELEARNING, epoch, batch, accuracy, HDF5_FILE_EXTENSION)
    print("Loading model from", weights_filename)
    model = load_model(weights_filename)
    print("Model loaded")

    vector_eval_function = K.function([model.layers[0].input], [model.layers[1].output])
    window_size = model.layers[0].input_shape[1]
    input_tensor = np.array(list(dictionary.values()))
    input_tensor.resize((math.ceil(len(dictionary) / window_size), window_size))
    embedding_size = model.layers[1].output_shape[2]
    return vector_eval_function([input_tensor])[0].reshape(-1, embedding_size)

def extract_save_emb(dataset, training_session, epoch, batch, accuracy):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)

    vectors = extract_emb(dataset, training_session, epoch, batch, accuracy)

    results = {}
    for word, word_num in dictionary.items():
        results[word] = vectors[int(word_num) - 1].tolist()

    result_filename = get_dataset_filename(dataset, ALL_FILENAME, EMB_POSTFIX, JSON_FILE_EXTENSION)
    save_json(result_filename, results)
    print("Embeddings saved at", result_filename)

extract_save_emb(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))