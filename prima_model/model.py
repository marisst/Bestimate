from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate
from keras.utils import plot_model

from prima_model.highway import highway_layers

# model hyperparameters
highway_layer_count = 20
lstm_nodes = 100
lstm_dropout = 0.1
lstm_recurrent_dropout = 0.1
final_dropout = 0.3
highway_activation = "relu" # relu or tanh

def create_model(max_text_length):
    
    text_input = Input(shape=(max_text_length, 300))
    masked_text_input = Masking()(text_input)

    context = LSTM(lstm_nodes)(masked_text_input)
    dense1 = Dense(12)(context)
    dense2 = Dense(12)(dense1)
    dense3 = Dense(6)(dense2)
    output = Dense(1)(dense3)

    model = Model(inputs=[text_input], outputs=[output])

    print("Model created")

    return model