import numpy as np

def mean_absolute_error(dataset, value):
    return np.average(np.absolute(dataset - value))

def mean_and_median(dataset):
    return (np.mean(dataset), np.median(dataset))


