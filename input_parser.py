import constants
import os

def select_datasets(datasets):

    # selecting all datasets
    if len(datasets) == 0:
        return [entry for entry in os.listdir(constants.DATA_FOLDER) if os.path.isdir("%s/%s" % (constants.DATA_FOLDER, entry))]

    for dataset in datasets:
        if not os.path.isdir("%s/%s" % (constants.DATA_FOLDER, dataset)):
            print("Dataset %s does not exist" % dataset)
            datasets.remove(dataset)

    return datasets