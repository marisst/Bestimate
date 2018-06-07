from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate, Activation, ActivityRegularization, Average
from keras.utils import plot_model
from keras.initializers import glorot_uniform

from training.highway import highway_layers
from utilities.constants import *


def create_deep_dense(hidden_unit_counts, previous_layer):

    for hidden_unit_count in hidden_unit_counts:
        previous_layer = Dense(hidden_unit_count)(previous_layer)
    return previous_layer


def create_field_context(max_text_length, embedding_size, lstm_node_count, lstm_recurrent_dropout, lstm_dropout):

    text_input = Input(shape=(max_text_length, embedding_size))
    masked_text_input = Masking()(text_input)

    kernel_initializer = glorot_uniform(seed=7)
    field_context = LSTM(
        lstm_node_count,
        dropout=lstm_dropout,
        recurrent_dropout=lstm_recurrent_dropout,
        kernel_initializer=kernel_initializer)(masked_text_input)

    return text_input, field_context


def create_model(max_text_length, embedding_size, model_params):

    if model_params["lstm_count"] == 2:

        summary_input, summary_context = create_field_context(
            max_text_length[0],
            embedding_size,
            model_params['lstm_node_count'],
            model_params['lstm_recurrent_dropout_1'],
            model_params['lstm_dropout_1'])

        description_input, description_context = create_field_context(
            max_text_length[1],
            embedding_size,
            model_params['lstm_node_count'],
            model_params['lstm_recurrent_dropout_2'],
            model_params['lstm_dropout_2'])
        
        context = Average()([summary_context, description_context])

    else:

        text_input, context = create_field_context(
            max_text_length[0],
            embedding_size,
            model_params['lstm_node_count'],
            model_params['lstm_recurrent_dropout'],
            model_params['lstm_dropout'])
    
    kernel_initializer = glorot_uniform(seed=7)
    highway = highway_layers(
        context,
        model_params['highway_layer_count'],
        kernel_initializer,
        activation=model_params['highway_activation'])
    drop = Dropout(model_params['dropout'])(highway)
    estimate = Dense(1, kernel_initializer=kernel_initializer)(drop)
    inputs = [text_input] if model_params["lstm_count"] == 1 else [summary_input, description_input]
    model = Model(inputs=inputs, outputs=[estimate])

    print("Model created")
    return model