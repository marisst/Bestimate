import numpy as np
from keras import losses
from keras import optimizers
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import load_model
import matplotlib.pyplot as plt
import calculate_baselines as bsl
import graph_helpers as gph
import load_data as data
import arrange_data as arr
import model as mdl
import save_results as save

# data parameters
dataset_count = 9
project_count = 374
max_summary_length = 30
max_description_length = 200

# training parameters
split_percentage = 75
learning_rate = 0.02
epochs = 1000
batch_size = 50

# load and arrange data
data = data.load(dataset_count, project_count, max_description_length)
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
model = mdl.create_model(dataset_count, project_count, max_summary_length, max_description_length)
rmsprop = optimizers.RMSprop(lr=learning_rate)
model.compile(loss=losses.mean_absolute_error, optimizer=rmsprop)
print(model.summary())

#preload weights
init_model = load_model("../weights/median-init-weights.hdf5")
model.set_weights(init_model.get_weights())

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
directory_name = "../weights/Andre"
save_weights = ModelCheckpoint(directory_name + "/Andre-{epoch:04d}-{val_loss:.0f}.hdf5")
save_results = LambdaCallback(on_epoch_end=lambda epoch, logs: save.save_logs(directory_name, epoch, logs))

# train and validate
callbacks = [save_weights, save_results, update_testing_line, update_training_line]
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, callbacks=callbacks)

# Save the model
model.save("lstm_model.h5")

plt.show()