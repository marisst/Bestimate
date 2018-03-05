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

def get_one_hot(category_ids, category_count):

    datapoint_count = len(category_ids)
    categories = np.unique(category_ids)
    
    one_hot = np.zeros((datapoint_count, category_count), dtype=np.bool_)
    for i, category_id in enumerate(category_ids):
        category_index = np.where(categories == category_id)[0][0]
        one_hot[i][category_index] = 1
    print("converted to one hot vectors")
    return one_hot

def load(dataset_count, project_count, max_description_length):

    limit = 39030
    embedding_size = 384
    partition_count = 10
    partition_size = limit // 384

    y = get_from_file("../preprocessed-data/y.pkl.gz", limit)

    raw_dataset_ids = get_from_file("../preprocessed-data/dataset-ids.pkl.gz", limit)
    dataset_ids = get_one_hot(raw_dataset_ids, dataset_count)
    raw_dataset_ids = None

    raw_project_ids = get_from_file("../preprocessed-data/project-ids.pkl.gz", limit)
    project_ids = get_one_hot(raw_project_ids, project_count)
    raw_project_ids = None

    summary_vectors = get_from_file("../preprocessed-data/summary-vectors.pkl.gz", limit)

    description_vectors = np.empty((limit, max_description_length, embedding_size))

    for i in range(partition_count):
        loaded_vectors = get_from_file("../preprocessed-data/description-vectors_%d.pkl.gz" % (i + 1))
        partition_start_index = i * partition_count
        for j in range(partition_size):
            k = partition_start_index + j
            description_vectors[k] = loaded_vectors[j]
        loaded_vectors = None

    return (dataset_ids, project_ids, summary_vectors, description_vectors, y)
