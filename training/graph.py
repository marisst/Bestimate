from utilities.constants import *
from embedding_pretraining.train_gensim import train_gensim
from functools import partial
from training import load_data as load
from training.graph_helpers import plot_losses, create_prediction_scatter
import multiprocessing
from keras.models import load_model

### Fake script, needs to get removed

def gensim_lookup(word_vectors, word):

    if word not in word_vectors:
        return None
    
    return word_vectors.get_vector(word)

weigths_directory_name = "results/23_ap_gensim/manual"

gensim_model = train_gensim(
        "ap",
        "skip-gram",
        469,
        11,
        7, 
        11,
        None,
        save=False,
        workers=2)
lookup = partial(gensim_lookup, gensim_model.wv)
del gensim_model

data, vector_dictionary = load.load_and_arrange(
    "ap",
    (60,20),
    None,
    (100,0),
    lookup)
x_train, y_train, x_test, y_test, x_valid, y_valid = data

best_model_filename = weigths_directory_name + "/model.h5"
best_model = load_model(best_model_filename)

model_params = {
    "lstm_count": "0",
    "lstm_node_count": 99,
    "conform_type": "hway",
    "conform_layer_count": 104,
    "conform_activation": "tanh",
    "dropout": 0.626346855852103,
    "batch_size": 512,
    "loss": "mean_absolute_error",
    "workers": 2,
    "optimizer": [
        "adam",
        0.0025851354140512027
    ],
    "max_words": [
        100,
        0
    ],
    "lstm_recurrent_dropout": 0.4665387396053841,
    "lstm_dropout": 0.602105113594757
}

for x, y, filename, title in [
    (x_test, y_test, "%s/%s%s" % (weigths_directory_name, "test_pred", PNG_FILE_XTENSION), "Testing dataset predictions"),
    (x_valid, y_valid, "%s/%s%s" % (weigths_directory_name, "val_pred", PNG_FILE_XTENSION), "Validation dataset predictions")]:
    try:
        create_prediction_scatter(best_model, x, y, filename, title, model_params, vector_dictionary)
    except multiprocessing.pool.MaybeEncodingError:
        # ON OS X when data larger than 4 GB
        continue
    except OverflowError:
        continue