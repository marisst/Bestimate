from sklearn.manifold import TSNE
import sys

from pretrain.extract_emb import extract_emb
from utilities.constants import *
from utilities.load_data import load_json, save_json

# https://github.com/keras-team/keras/issues/5204

def extract_2dim_embeddings(dataset, training_session, epoch, batch, accuracy):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)

    vectors = extract_emb(dataset, training_session, epoch, batch, accuracy)
    vectors_2dim = TSNE(n_components=2, verbose=1, perplexity=100, n_iter=1000).fit_transform(vectors)

    results = {}
    for word, word_num in dictionary.items():
        results[word] = (vectors_2dim[word_num][0].item(), vectors_2dim[word_num][1].item())

    result_filename = get_dataset_filename(dataset, ALL_FILENAME, EMB2DIM_POSTFIX, JSON_FILE_EXTENSION)
    save_json(result_filename, results)
    print("Embeddings reduced to two dimensions with t-SNE and saved at", result_filename)

extract_2dim_embeddings(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]))