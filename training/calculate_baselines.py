import numpy as np

MEAN_HUMAN_ABSOLUTE_PERCENTAGE_ERROR = 30

def mean_absolute_error(dataset, value):
    return np.mean(np.abs(dataset - value))

def mean_absolute_percentage_error(dataset, value):
    return np.mean(np.abs((dataset - value) / dataset)) * 100

def mean_squared_error(dataset, value):
    return np.mean(np.square(dataset - value))

def mean_and_median(dataset):
    return (np.mean(dataset), np.median(dataset))

def mean_human_absolute_error(dataset):
    return sum([(label * MEAN_HUMAN_ABSOLUTE_PERCENTAGE_ERROR) / (len(dataset) * 100) for label in dataset])
