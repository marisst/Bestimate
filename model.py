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

def create_model(max_text_length):
    
    text_input = Input(shape=(max_text_length, 384))
    masked_text_input = Masking()(text_input)

    context = LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(masked_text_input)

    highway_layer = highway_layers(context, highway_layer_count, highway_activation)
    drop = Dropout(final_dropout)(highway_layer)
    output = Dense(1)(drop)

    model = Model(inputs=[text_input], outputs=[output])
    #plot_model(model, to_file='model.png')

    print("Model created")

    return model