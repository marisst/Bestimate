import os

from utilities.constants import *

def select_datasets(datasets):

    # selecting all datasets
    if len(datasets) == 0:
        return [entry for entry in os.listdir(DATA_FOLDER) if os.path.isdir("%s/%s" % (DATA_FOLDER, entry))]

    for dataset in datasets:
        if not os.path.isdir("%s/%s" % (DATA_FOLDER, dataset)):
            print("Dataset %s does not exist" % dataset)
            datasets.remove(dataset)

    return datasets