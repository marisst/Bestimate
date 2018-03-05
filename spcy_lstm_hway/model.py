from keras.models import Model, load_model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, concatenate
from highway import highway_layers
from keras.utils import plot_model

# model hyperparameters
highway_layer_count = 20
lstm_nodes = 10
lstm_dropout = 0.1
lstm_recurrent_dropout = 0.1
final_dropout = 0.3
highway_activation = "relu" # relu or tanh

def create_model(dataset_count, project_count, max_summary_length, max_description_length):
    
    summary_input = Input(shape=(max_summary_length, 384))
    masked_summary_input = Masking()(summary_input)

    description_input = Input(shape=(max_description_length, 384))
    masked_description_input = Masking()(description_input)

    summary_context = LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(masked_summary_input)
    description_context = LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(masked_description_input)

    dataset_input = Input(shape=(dataset_count,))
    project_input = Input(shape=(project_count,))

    concatenated = concatenate([dataset_input, project_input, summary_context, description_context])

    highway_layer = highway_layers(concatenated, highway_layer_count, highway_activation)
    drop = Dropout(final_dropout)(highway_layer)
    output = Dense(1)(drop)

    model = Model(inputs=[dataset_input, project_input, summary_input, description_input], outputs=[output])

    print("Model created")

    return model
