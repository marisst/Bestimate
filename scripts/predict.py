import numpy as np
import spacy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras import losses
import xml.etree.ElementTree as et
import time
import logging
import codecs, json
from keras.models import load_model
import json
import csv
import matplotlib.pyplot as plt

import load_data as data

# fix random seed for reproducibility
np.random.seed(7)

split_percentage = 75
max_summary_length = 30
max_description_length = 200

#tree = et.parse('./full_consolidated.xml')
#print("XML parsing finished")
#root = tree.getroot()

skip_records = 0
process_records = 39031

y = np.zeros((process_records))
i = 0
skipped = 0

x = np.zeros((process_records, max_summary_length, 384))

#texts = []
#for item in root:
#    summary = item.find('summary').text
#    time_spent_in_seconds = int(item.find('seconds').text)
#
#    if time_spent_in_seconds < 600 or time_spent_in_seconds > 101241:
#        continue
#
#    if skipped < skip_records:
#        skipped = skipped + 1
#        continue
#
#    texts.append((summary, time_spent_in_seconds))
#texts = np.array(texts)

dataset_ids, project_ids, summary_vectors, description_vectors, y = data.load()

split = len(y) * split_percentage // 100

# shuffle
permutation = np.random.permutation(len(y))
dataset_ids = dataset_ids[permutation]
project_ids = project_ids[permutation]
summary_vectors = summary_vectors[permutation]
description_vectors = description_vectors[permutation]
y = y[permutation]
print("Shuffled")

model = load_model("models-natt/weights-0118-12949.hdf5")

predictions = model.predict([dataset_ids, project_ids, summary_vectors, description_vectors])

y = y / 3600
predictions = predictions / 3600

#results = []
deviations = []
for i, prediction in enumerate(predictions):
    #comp = int(texts[i, 1]) / 3600
    #if comp != y[i]:
    #    print ("Error %d != %d" % (comp, y[i]))
    #results.append([texts[i, 0], y[i], prediction[0], prediction[0] - y[i]])
    deviations.append(abs(prediction[0] - y[i]))
#results = sorted(results, key=lambda x: abs(int(x[3])))

#with open("predictions.csv", "w", newline="", encoding="utf-8-sig") as file:
#    w = csv.writer(file)
#    for result in results:
#        try:
#            w.writerow([result[0], "%.2f" % (result[1]), "%.2f" % (result[2]), "%.2f" % (result[3])])
#        except UnicodeEncodeError:
#            continue

ax1 = plt.subplot2grid((1, 2), (0, 0))
ax2 = plt.subplot2grid((1, 2), (0, 1))

max_plot_hours = 28

ax1.set_title("Training dataset predictions")
ax1.axis([0, max_plot_hours, 0, max_plot_hours])
ax1.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
ax1.set_xlabel("Time spent, hours", fontsize=10)
ax1.set_ylabel("Model estimate, hours", fontsize=10)
ax1.scatter(y[:split], predictions[:split], c=deviations[:split], cmap='coolwarm_r')

ax2.set_title("Testing dataset predictions")
ax2.axis([0, max_plot_hours, 0, max_plot_hours])
ax2.plot([0, max_plot_hours], [0, max_plot_hours], 'r--')
ax2.set_xlabel("Time spent, hours", fontsize=10)
ax2.set_ylabel("Model estimate, hours", fontsize=10)
ax2.scatter(y[split:], predictions[split:], c=deviations[split:], cmap='coolwarm_r')


plt.show()