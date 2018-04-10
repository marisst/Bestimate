from constants import *
import numpy as np
from keras import losses
from keras import optimizers
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import calculate_baselines as bsl
import graph_helpers as gph
import load_data
import arrange_data as arr
import model as mdl
import save_results as save
import sys

# training parameters
split_percentage = 75
learning_rate = 0.02
epochs = 1000
batch_size = 50

def train_on_dataset(dataset):

    # load and arrange data
    filename = get_vectorized_dataset_filename(dataset)
    data = load_data.load_pickle(filename)
    shuffled_data = arr.shuffle(data)
    splitted_data = arr.split_train_test(shuffled_data, split_percentage)
    x_train, y_train, x_test, y_test = splitted_data

    #calculate baseline losses
    train_mean, train_median = bsl.mean_and_median(y_train)
    mean_baseline = bsl.mean_absolute_error(y_test, train_mean)
    median_baseline = bsl.mean_absolute_error(y_test, train_median)

    # weight initialization with median value
    #y_train = np.full(y_train.shape, train_median)

    # create model
    max_text_length = x_test.shape[1]
    model = mdl.create_model(max_text_length)
    rmsprop = optimizers.RMSprop(lr=learning_rate)
    model.compile(loss=losses.mean_absolute_error, optimizer=rmsprop)
    print(model.summary())

    #preload weights
    #init_model = load_model("weights/median-init-weights.hdf5")
    #model.set_weights(init_model.get_weights())

    # create training graph
    plt.ion()
    axs = plt.gca()
    gph.create_losses_plot(axs, "Training and testing losses", epochs, (3, 5.2), mean_baseline, median_baseline)
    training_line, = axs.plot([], [], 'b-')
    testing_line, = axs.plot([], [], 'g-')
    plt.draw_all()
    plt.pause(5)

    # update graph lines after every epoch
    update_training_line = LambdaCallback(on_epoch_end=lambda epoch, logs: gph.update_line(training_line, epoch, logs['loss']))
    update_testing_line = LambdaCallback(on_epoch_end=lambda epoch, logs: gph.update_line(testing_line, epoch, logs['val_loss']))

    # save weights and results after every epoch
    load_data.create_folder_if_needed(WEIGTHS_FOLDER)
    training_session_name = load_data.get_next_dataset_name(WEIGTHS_FOLDER)
    weigths_directory_name = get_weigths_folder_name(dataset, training_session_name)
    load_data.create_folder_if_needed(weigths_directory_name)
    weigths_filename = get_weigths_filename(dataset, training_session_name)
    save_weights = ModelCheckpoint(weigths_filename)
    save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(weigths_directory_name, epoch, logs))

    # train and validate
    callbacks = [save_weights, save_results, update_testing_line, update_training_line]
    model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

    # Save the model
    model.save(weigth_directory_name + "/lstm_model.h5")

    plt.show()

train_on_dataset(sys.argv[1])