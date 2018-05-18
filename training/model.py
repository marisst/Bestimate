from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate, Activation, ActivityRegularization
from keras.utils import plot_model
from keras.regularizers import l2

from training.highway import highway_layers

def create_deep_dense(hidden_unit_counts, previous_layer):

    for hidden_unit_count in hidden_unit_counts:
        previous_layer = Dense(hidden_unit_count)(previous_layer)
    return previous_layer

def create_model(max_text_length, embedding_size, model_params):
    
    text_input = Input(shape=(max_text_length, embedding_size))
    masked_text_input = Masking()(text_input)
    context = LSTM(model_params['lstm_node_count'], dropout=model_params['lstm_dropout'], recurrent_dropout=model_params['lstm_recurrent_dropout'], activity_regularizer=l2(0.00001))(masked_text_input)
    highway = highway_layers(context, model_params['highway_layer_count'], activation=model_params['highway_activation'], kernel_regularizer=l2(0.00001))
    drop = Dropout(model_params['dropout'])(highway)
    estimate = Dense(1)(drop)
    model = Model(inputs=[text_input], outputs=[estimate])

    print("Model created")

    return model