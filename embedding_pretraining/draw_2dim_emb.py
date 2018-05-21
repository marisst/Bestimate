import sys
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np

from utilities.constants import *
from utilities.load_data import load_json

# https://github.com/keras-team/keras/issues/5204

def draw_embeddings(dataset, skip_words, show_words):

    filename = get_dataset_filename(dataset, ALL_FILENAME, EMB2DIM_POSTFIX, JSON_FILE_EXTENSION)
    embeddings = load_json(filename)

    color_map = plt.cm.get_cmap("RdYlGn")
    figure = plt.figure(figsize=(10, 10))
    ax = plt.subplot(aspect="equal", )

    coordinates = np.array(list(embeddings.values()))
    scatter = ax.scatter(coordinates[:,0], coordinates[:,1], lw=0, s=10, cmap=color_map)
    plt.xlim(70)
    plt.ylim(70)
    ax.axis("off")
    ax.axis("tight")

    for i, word in enumerate(embeddings):

        if i < skip_words:
            continue

        if i > skip_words + show_words:
            break

        ax.annotate(word, embeddings[word])

    plt.show()

draw_embeddings(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))