import gzip
import pickle
import numpy as np

def get_from_file(file_name, limit = 0):
    with gzip.open(file_name, "rb") as file:
        object = pickle.load(file)
        if limit != 0:
            object = object[:limit]
        print("%s loaded" % file_name)
        return object

def get_one_hot(category_ids):

    datapoint_count = len(category_ids)
    categories = np.unique(category_ids)
    category_count = len(categories)
    
    one_hot = np.zeros((datapoint_count, category_count), dtype=np.bool_)
    for i, category_id in enumerate(category_ids):
        category_index = np.where(categories == category_id)[0][0]
        one_hot[i][category_index] = 1
    print("converted to one hot vectors")
    return one_hot

def load():

    limit = 3903
    embedding_size = 384
    partition_count = 1

    y = get_from_file("preprocessed-data/y.pkl.gz", limit)

    raw_dataset_ids = get_from_file("preprocessed-data/dataset-ids.pkl.gz", limit)
    dataset_ids = get_one_hot(raw_dataset_ids)

    raw_project_ids = get_from_file("preprocessed-data/project-ids.pkl.gz", limit)
    project_ids = get_one_hot(raw_project_ids)

    summary_vectors = get_from_file("preprocessed-data/summary-vectors.pkl.gz", limit)

    raw_description_vectors = [get_from_file("preprocessed-data/description-vectors_%d.pkl.gz" % (i + 1)) for i in range(partition_count)]
    description_vectors = np.array(raw_description_vectors)
    description_vectors.flatten()

    return (dataset_ids, project_ids, summary_vectors, description_vectors, y)
