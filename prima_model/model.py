from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate, Activation
from keras.utils import plot_model

from prima_model.highway import highway_layers

def create_deep_dense(hidden_unit_counts, previous_layer):

    for hidden_unit_count in hidden_unit_counts:
        previous_layer = Dense(hidden_unit_count)(previous_layer)
    return previous_layer

def create_model(max_text_length, embedding_size):
    
    text_input = Input(shape=(max_text_length, embedding_size))
    masked_text_input = Masking()(text_input)
    context = LSTM(150)(masked_text_input)
    highway = highway_layers(context, 50, activation="relu")
    prediction = Dense(1)(highway)
    output = Activation("relu")(prediction)
    model = Model(inputs=[text_input], outputs=[output])

    print("Model created")

    return model