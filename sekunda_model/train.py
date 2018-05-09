import sys
from sekunda_model.load_data import load_data
from sekunda_model.model import create_model
from pretrain.load_data import get_vocabulary_size

MAX_SENTENCE_LENGTH = 150
LSTM_NODES = 50

def train_on_dataset(dataset):

    x, y = load_data(dataset, MAX_SENTENCE_LENGTH)
    vocabulary_size = get_vocabulary_size(dataset)

    model = create_model(x.shape[1], MAX_SENTENCE_LENGTH, vocabulary_size, LSTM_NODES)

train_on_dataset(sys.argv[1])