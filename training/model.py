from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate, Activation, ActivityRegularization, Average
from keras.utils import plot_model
from keras.regularizers import l1, l2, l1_l2
from keras.initializers import glorot_uniform

from training.highway import highway_layers
from utilities.constants import *

def create_deep_dense(hidden_unit_counts, previous_layer):

    for hidden_unit_count in hidden_unit_counts:
        previous_layer = Dense(hidden_unit_count)(previous_layer)
    return previous_layer

def create_model(max_words, embedding_size, model_params):
    
    summary_input = Input(shape=(max_words, embedding_size))
    masked_summary_input = Masking()(summary_input)

    description_input = Input(shape=(max_words, embedding_size))
    masked_description_input = Masking()(description_input)

    reg = {}
    for regularizer_name in REGULARIZERS:
        if model_params[regularizer_name + "-regularizer-l1"][0] == True and model_params[regularizer_name + "-regularizer-l2"][0] == True:
            reg[regularizer_name] = l1_l2(model_params[regularizer_name + "-regularizer-l1"][1], model_params[regularizer_name + "-regularizer-l2"][1])
        elif model_params[regularizer_name + "-regularizer-l1"][0] == True:
            reg[regularizer_name] = l1(model_params[regularizer_name + "-regularizer-l1"][1])
        elif model_params[regularizer_name + "-regularizer-l2"][0] == True:
            reg[regularizer_name] = l2(model_params[regularizer_name + "-regularizer-l2"][1])
        else:
            reg[regularizer_name] = None

    kernel_initializer = glorot_uniform(seed=7)
    summary_context = LSTM(
        model_params['lstm_node_count'],
        dropout=model_params['summary_lstm_dropout'],
        recurrent_dropout=model_params['summary_lstm_recurrent_dropout'],
        activity_regularizer=reg['summary-lstm-activity'],
        kernel_initializer=kernel_initializer)(masked_summary_input)

    description_context = LSTM(
        model_params['lstm_node_count'],
        dropout=model_params['description_lstm_dropout'],
        recurrent_dropout=model_params['description_lstm_recurrent_dropout'],
        activity_regularizer=reg['description-lstm-activity'],
        kernel_initializer=kernel_initializer)(masked_description_input)

    merged_context = Average()([summary_context, description_context])
        
    highway = highway_layers(
        merged_context,
        model_params['highway_layer_count'],
        kernel_initializer,
        activation=model_params['highway_activation'])
    drop = Dropout(model_params['dropout'])(highway)

    act_l1 = model_params["activity-regularizer-l1"][1] if model_params["activity-regularizer-l1"][0] == True else 0
    act_l2 = model_params["activity-regularizer-l2"][1] if model_params["activity-regularizer-l2"][0] == True else 0
    act = ActivityRegularization(l1=act_l1, l2=act_l2)(drop)
    estimate = Dense(1, kernel_initializer=kernel_initializer)(act)
    model = Model(inputs=[summary_input, description_input], outputs=[estimate])

    print("Model created")

    return model