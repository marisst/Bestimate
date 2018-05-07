from keras import backend as K
from keras.models import load_model
import numpy as np
from sklearn.manifold import TSNE
import sys
import math

from utilities.constants import *
from utilities.load_data import load_json, save_json

# https://github.com/keras-team/keras/issues/5204

def extract_embeddings(dataset, training_session, epoch, accuracy):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)

    weights_filename = "%s/%s-%s_%s/weights-%04d-%.2f%s" % (WEIGTHS_FOLDER, dataset, training_session, PRELEARNING, epoch, accuracy, HDF5_FILE_EXTENSION)
    model = load_model(weights_filename)
    print("Model loaded")

    vector_eval_function = K.function([model.layers[0].input], [model.layers[1].output])
    window_size = model.layers[0].input_shape[1]
    input_tensor = np.array(list(dictionary.values()))
    input_tensor.resize((math.ceil(len(dictionary) / window_size), window_size))
    vectors = vector_eval_function([input_tensor])[0].reshape(-1, 10)

    vectors_2dim = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000).fit_transform(vectors)

    results = {}
    for word, word_num in dictionary.items():
        results[word] = (vectors_2dim[word_num][0].item(), vectors_2dim[word_num][1].item())

    result_filename = get_dataset_filename(dataset, ALL_FILENAME, EMB2DIM_POSTFIX, JSON_FILE_EXTENSION)
    save_json(result_filename, results)
    print("Embeddings reduced to two dimensions with t-SNE and saved at", result_filename)

extract_embeddings(sys.argv[1], sys.argv[2], int(sys.argv[3]), float(sys.argv[4]))