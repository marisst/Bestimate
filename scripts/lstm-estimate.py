import numpy as np
from keras.models import Model, load_model
from keras.layers import Dense, Masking, LSTM, Input, Dropout, Concatenate
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import losses
from keras import optimizers
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import codecs, json
import csv
import load_data as data

###
import keras.backend as K
from keras.layers import Dense, Activation, Multiply, Add, Lambda
import keras.initializers
 
def highway_layers(value, n_layers, activation="tanh", gate_bias=-3):
    dim = K.int_shape(value)[-1]
    gate_bias_initializer = keras.initializers.Constant(gate_bias)
    for i in range(n_layers):     
        gate = Dense(units=dim, bias_initializer=gate_bias_initializer)(value)
        gate = Activation("sigmoid")(gate)
        negated_gate = Lambda(
            lambda x: 1.0 - x,
            output_shape=(dim,))(gate)
        transformed = Dense(units=dim)(value)
        transformed = Activation(activation)(value)
        transformed_gated = Multiply()([gate, transformed])
        identity_gated = Multiply()([negated_gate, value])
        value = Add()([transformed_gated, identity_gated])
    return value
###

def mean_median_absolute_error(dataset):
    median = np.median(dataset)
    return np.average(np.absolute(dataset - median))

def mean_mean_absolute_error(dataset):
    mean = np.mean(dataset)
    return np.average(np.absolute(dataset - mean))

def update_line(hl, epoch, loss):
    hl.set_xdata(np.append(hl.get_xdata(), [epoch]))
    hl.set_ydata(np.append(hl.get_ydata(), [(loss / 3600)]))
    plt.draw()
    plt.pause(0.1)

def update_graphs(model, axs, x):
    pred = model.predict(x)
    dev = []
    for i, p in enumerate(pred):
        dev.append(abs(p - x[i]))
    axs.clear()
    axs.scatter(x, pred, c=dev, cmap='coolwarm_r')
    plt.pause(0.1)
    plt.draw()

def save_to_csv(epoch, logs):
    with open("models-night/results.csv", "a", newline="", encoding="utf-8-sig") as resultFile:
        w.writerow([epoch, "%.4f" % logs['loss'], "%.4f" % logs['val_loss']])

# fix random seed for reproducibility
np.random.seed(7)

split_percentage = 75
max_summary_length = 30
max_description_length = 200

dataset_ids, project_ids, summary_vectors, description_vectors, y = data.load()

split = len(y) * split_percentage // 100

# shuffle
permutation = np.random.permutation(len(y))
summary_vectors = summary_vectors[permutation]
description_vectors = description_vectors[permutation]
y = y[permutation]
dataset_ids = dataset_ids[permutation]
project_ids = project_ids[permutation]

# divide in training and testing sets
summary_vectors_train = summary_vectors[:split]
description_vectors_train = description_vectors[:split]
y_train = y[:split]
dataset_ids_train = dataset_ids[:split]
project_ids_train = project_ids[:split]

summary_vectors_test = summary_vectors[split:]
description_vectors_test = description_vectors[split:]
y_test = y[split:]
dataset_ids_test = dataset_ids[split:]
project_ids_test = project_ids[split:]

#calculate median baseline results
train_median = np.median(y_train)
test_median = np.median(y_test)
train_mean = np.mean(y_train)
test_mean = np.mean(y_test)

print("Train median: %d" % train_median)
train_MMAAE = mean_median_absolute_error(y_train)
print("Train median average absolute error: %.4f" % train_MMAAE)

print("Test median: %d" % test_median)
test_MMAAE = mean_median_absolute_error(y_test)
print("Test median average absolute error: %.4f" % test_MMAAE)

print("Train mean: %d" % train_mean)
print("Train mean average absolute error: %.4f" % mean_mean_absolute_error(y_train))

print("Test mean: %d" % test_mean)
print("Test mean average absolute error: %.4f" % mean_mean_absolute_error(y_test))

# weight initialization with median value
# y_train = np.full(y_train.shape, train_median)

# hyperparameters
highway_layer_count = 20
lstm_nodes = 10
learning_rate = 0.01
lstm_dropout = 0.2
lstm_recurrent_dropout = 0.2
final_dropout = 0.5
highway_activation = "relu" # relu or tanh
epochs = 3000
lstm_factor = 1

# create model
summary_input = Input(shape=(max_summary_length, 384))
masked_summary_input = Masking()(summary_input)

description_input = Input(shape=(max_description_length, 384))
masked_description_input = Masking()(description_input)

dataset_input = Input(shape=(dataset_count,))
project_input = Input(shape=(project_count,))

summary_context = LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(masked_summary_input)
description_context = LSTM(lstm_nodes * lstm_factor, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout)(masked_description_input)

concatenated = keras.layers.concatenate([dataset_input, project_input, summary_context, description_context])

highway = highway_layers(concatenated, highway_layer_count, highway_activation)
drop = Dropout(final_dropout)(highway)
output = Dense(1)(drop)

model = Model(input=[dataset_input, project_input, summary_input, description_input], output=[output])

#preload weights
#init_model = load_model("weights/weights-full-init-median.hdf5")
#model.set_weights(init_model.get_weights())

rmsprop = optimizers.RMSprop(lr=learning_rate)
model.compile(loss=losses.mean_absolute_error, optimizer=rmsprop)
print(model.summary())

# save model after every epoch
saveCallback = keras.callbacks.ModelCheckpoint("models-natt/weights-{epoch:04d}-{val_loss:.0f}.hdf5", period=1)

# create graphs

max_plot_hours = 28

plt.ion()

ax1 = plt.subplot2grid((2, 2), (0, 0), colspan=2)
ax2 = plt.subplot2grid((2, 2), (1, 0))
ax3 = plt.subplot2grid((2, 2), (1, 1))

ax1.axis([0, epochs, 3, 5.2])
ax1.axhline(y=train_MMAAE / 3600, color='b', linestyle='dashed')
ax1.axhline(y=test_MMAAE / 3600, color='g', linestyle='dashed')

ax1.set_title("Training and testing losses")
ax1.set_ylabel("Mean absolute error, hours", fontsize=10)
ax1.set_xlabel("Epoch")
horizontal_padding = epochs * 0.003
vertical_padding = 0.05
ax1.text(horizontal_padding, (train_MMAAE / 3600) + vertical_padding, 'Training median loss', verticalalignment='bottom', horizontalalignment='left', color='b', fontsize=10)
ax1.text(horizontal_padding, (test_MMAAE / 3600) - vertical_padding, 'Testing median loss', verticalalignment='top', horizontalalignment='left', color='g', fontsize=10)

training_line, = ax1.plot([], [], 'b-')
testing_line, = ax1.plot([], [], 'g-')

ax2.set_title("Training dataset predictions")
ax2.axis([0, max_plot_hours, 0, max_plot_hours])
ax2.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
ax2.set_ylabel("Model estimate, hours", fontsize=10)
ax2.set_xlabel("Time spent, hours", fontsize=10)

ax3.set_title("Testing dataset predictions")
ax3.axis([0, max_plot_hours, 0, max_plot_hours])
ax3.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
ax3.set_ylabel("Model estimate, hours", fontsize=10)
ax3.set_xlabel("Time spent, hours", fontsize=10)

plt.draw_all()
plt.pause(10)

# print model graphs after every epoch
update_training_line = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: update_line(training_line, epoch, logs['loss']))
update_testing_line = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: update_line(testing_line, epoch, logs['val_loss']))
update_train_graph = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: update_graphs(model, ax2, [dataset_ids_train, project_ids_train, summary_vectors_train, description_vectors_train]))
update_test_graph = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: update_graphs(model, ax3, [dataset_ids_test, project_ids_test, summary_vectors_test, description_vectors_test]))

#CSV
result_file = open("models-natt/results.csv", "a", newline="", encoding="utf-8-sig")
w = csv.writer(result_file)
save_csv_l = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: save_to_csv(epoch, logs))

# train and validate
model.fit([dataset_ids_train, project_ids_train, summary_vectors_train, description_vectors_train], y_train, validation_data=([dataset_ids_test, project_ids_test, summary_vectors_test, description_vectors_test], y_test), epochs=epochs, batch_size=50, callbacks=[saveCallback, save_csv_l, update_testing_line, update_training_line])

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=1)
print(scores)

# Save the model
model.save("lstm_model.h5")

result_file.close()
plt.show()