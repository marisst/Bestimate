import numpy as np

def shuffle(data):

    # fix random seed for reproducibility
    np.random.seed(7)

    x, y = data
    permutation = np.random.permutation(len(y))

    shuffled_x = x[permutation]
    x = None

    shuffled_y = y[permutation]
    y = None

    print("Data shuffled")

    return (shuffled_x, shuffled_y)
    
def split(data, split_index):
    return (data[:split_index], data[split_index:])

def split_train_test(data, split_percentage):

    x, y = data
    split_index = len(y) * split_percentage // 100

    x_train, x_test = split(x, split_index)
    y_train, y_test = split(y, split_index)
    y = None

    print("Data splitted in training and testing sets")

    return (x_train, y_train, x_test, y_test)