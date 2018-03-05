import numpy as np

def shuffle(data):

    # fix random seed for reproducibility
    np.random.seed(7)

    dataset_ids, project_ids, summary_vectors, description_vectors, y = data
    permutation = np.random.permutation(len(y))

    shuffled_dataset_ids = dataset_ids[permutation]
    dataset_ids = None

    shuffled_project_ids = project_ids[permutation]
    project_ids = None

    shuffled_summary_vectors = summary_vectors[permutation]
    summary_vectors = None

    shuffled_description_vectors = description_vectors[permutation]
    description_vectors = None

    shuffled_y = y[permutation]

    print("Shuffled")

    return (shuffled_dataset_ids, shuffled_project_ids, shuffled_summary_vectors, shuffled_description_vectors, shuffled_y)
    
def split(data, split_index):
    return (data[:split_index], data[split_index:])

def split_train_test(data, split_percentage):

    dataset_ids, project_ids, summary_vectors, description_vectors, y = data
    split_index = len(y) * split_percentage // 100

    dataset_ids_train, dataset_ids_test = split(dataset_ids, split_index)
    dataset_ids = None

    project_ids_train, project_ids_test = split(project_ids, split_index)
    project_ids = None

    summary_vectors_train, summary_vectors_test = split(summary_vectors, split_index)
    summary_vectors = None

    description_vectors_train, description_vectors_test = split(description_vectors, split_index)
    description_vectors = None

    y_train, y_test = split(y, split_index)
    y = None

    x_train = [dataset_ids_train, project_ids_train, summary_vectors_train, description_vectors_train]
    x_test = [dataset_ids_test, project_ids_test, summary_vectors_test, description_vectors_test]

    print("Splitted in training and testing sets")

    return (x_train, y_train, x_test, y_test)

