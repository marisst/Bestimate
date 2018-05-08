from keras.models import Model
from keras.layers import Dense, Input, LSTM, Masking
from prima_model.highway import highway_layers

def create_model(embedding_size, max_sentence_length, vocabulary_size, lstm_nodes):

    input_layer = Input(shape=(embedding_size, max_sentence_length,), name='main_input')
    masked_input_layer = Masking(mask_value=0)(input_layer)
    context_layer = LSTM(lstm_nodes)(masked_input_layer)
    highway_layer = highway_layers(context_layer, 20, "relu")
    estimate_layer = Dense(1)(highway_layer)
    model = Model(inputs=[input_layer], outputs=[estimate_layer])

    print("Model created")
    return model
