import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import graph_helpers as graphs
import load_data as data
import arrange_data as arr

dataset_count = 9
project_count = 374
max_summary_length = 30
max_description_length = 200

split_percentage = 75
max_plot_hours = 28

data = data.load(dataset_count, project_count, max_description_length)
shuffled_data = arr.shuffle(data)
dataset_ids, project_ids, summary_vectors, description_vectors, y = shuffled_data
split = len(y) * split_percentage // 100
model = load_model("../weights/median-init-weights.hdf5")
predictions = model.predict([dataset_ids, project_ids, summary_vectors, description_vectors])

dataset_ids = None
project_ids = None
summary_vectors = None
description_vectors = None

y = y / 3600
predictions = predictions / 3600
deviations = [abs(prediction[0] - y[i]) for i, prediction in enumerate(predictions)]

ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

graphs.create_prediction_scatter(ax1, "Training dataset predictions", max_plot_hours)
ax1.scatter(y[:split], predictions[:split], c=deviations[:split], cmap='coolwarm_r', marker='x', alpha = 0.5)

graphs.create_prediction_scatter(ax2, "Testing dataset predictions", max_plot_hours)
ax2.scatter(y[split:], predictions[split:], c=deviations[split:], cmap='coolwarm_r', marker='x', alpha = 0.5)

plt.show()