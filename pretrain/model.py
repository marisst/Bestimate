from keras.layers import Dense, Embedding, Input, LSTM
from keras.models import Model
from keras.activations import softmax

# model hyperparameters
lstm_nodes = 50

def create_model(window_size, embedding_size, vocabulary_size):

    input_layer = Input(shape=(window_size,), dtype='int32', name='main_input')
    embedding_layer = Embedding(output_dim=embedding_size, input_dim=vocabulary_size, input_length=window_size)(input_layer)
    context_layer = LSTM(lstm_nodes)(embedding_layer)
    dense_layer = Dense(vocabulary_size, activation=softmax)(context_layer)

    model = Model(inputs=[input_layer], outputs=[dense_layer])

    print("Model created")

    return model