import matplotlib.pyplot as plt
from utilities import load_data, constants.*
from utilities.constants import *
import sys
import numpy as np

def show_histogram(dataset):

    filename = get_vectorized_dataset_filename(dataset)
    x, y = load_data.load_pickle(filename)

    if y is None:
        return 

    max_hours = int(input("Please input the maximum number of hours to display in the histogram: "))
    hist = plt.hist(y / SECONDS_IN_HOUR, bins = max_hours * 12, range = (0, max_hours - 1 / SECONDS_IN_HOUR))
    plt.xticks(np.arange(0, max_hours + 1, 1))
    plt.show(hist)

show_histogram(sys.argv[1])