from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate, Activation, ActivityRegularization
from keras.utils import plot_model
from keras.regularizers import l2

from prima_model.highway import highway_layers

def create_deep_dense(hidden_unit_counts, previous_layer):

    for hidden_unit_count in hidden_unit_counts:
        previous_layer = Dense(hidden_unit_count)(previous_layer)
    return previous_layer

def create_model(max_text_length, embedding_size):
    
    text_input = Input(shape=(max_text_length, embedding_size))
    masked_text_input = Masking()(text_input)
    context = LSTM(10, dropout=0.1, recurrent_dropout=0.1, activity_regularizer=l2(0.0001))(masked_text_input)
    highway = highway_layers(context, 20, activation="relu", kernel_regularizer=l2(0.0001))
    drop = Dropout(0.2)(highway)
    estimate = Dense(1)(drop)
    model = Model(inputs=[text_input], outputs=[estimate])

    print("Model created")

    return model