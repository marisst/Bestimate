from keras.models import Model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate, Activation, ActivityRegularization
from keras.utils import plot_model
from keras.regularizers import l1, l2, l1_l2

from training.highway import highway_layers
from utilities.constants import *

def create_deep_dense(hidden_unit_counts, previous_layer):

    for hidden_unit_count in hidden_unit_counts:
        previous_layer = Dense(hidden_unit_count)(previous_layer)
    return previous_layer

def create_model(max_text_length, embedding_size, model_params):
    
    text_input = Input(shape=(max_text_length, embedding_size))
    masked_text_input = Masking()(text_input)

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

    context = LSTM(
        model_params['lstm_node_count'],
        dropout=model_params['lstm_dropout'],
        recurrent_dropout=model_params['lstm_recurrent_dropout'],
        activity_regularizer=reg['lstm-activity'])(masked_text_input)
    highway = highway_layers(
        context,
        model_params['highway_layer_count'],
        activation=model_params['highway_activation'])
    drop = Dropout(model_params['dropout'])(highway)

    act_l1 = model_params["activity-regularizer-l1"][1] if model_params["activity-regularizer-l1"][0] == True else 0
    act_l2 = model_params["activity-regularizer-l2"][1] if model_params["activity-regularizer-l2"][0] == True else 0
    act = ActivityRegularization(l1=act_l1, l2=act_l2)(drop)
    estimate = Dense(1)(act)
    model = Model(inputs=[text_input], outputs=[estimate])

    print("Model created")

    return model