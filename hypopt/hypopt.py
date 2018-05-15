from hyperopt import fmin, tpe, hp

space = {
    'data_selection' : hp.choice('data_selection', ['project', 'repository', 'cross-repository']),
    'min_word_count': hp.choice('min_word_count', [1, 4, 8, 16, 32, 64]),
    'min_time_spent': hp.choice('min_time_spent', [1, 4, 8, 16, 32]),
    'max_time_spent': hp.choice('max_time_spent', [240, 360, 480, 720, 960, 1920]),
    'min_project_size': hp.choice('min_project_size', [1, 32, 64, 128, 512, 1024]),
    'even_distribution': hp.choice('even_distribution', [True, False]),
    'word_embeddings': hp.choice('word_embeddings', [
        {
            "type": "spacy"
        },
        {
            "type": "gensim",
            "algorithm": hp.choice("word_embeddings_algorithm", ["skip-gram", "CBOW"]),
            "embedding_size": hp.choice("word_embeddings_embedding_size", [5, 10, 50, 100, 200, 300]),
            "minimum_count": hp.choice("word_embeddings_minimum_count", [1, 2, 4, 8, 16]),
            "window_size": hp.choice("word_embeddings_window_size", [3, 5, 7, 9, 11, 15]),
            "iterations": hp.choice("word_embeddings_iterations", [1, 3, 5, 7, 9])
        }
    ]),
    'maximum_word_count': hp.choice('maximum_word_count', [32, 64, 128, 256]),
    'lstm_node_count': hp.choice('lstm_node_count', [4, 8, 16, 32, 64, 128, 256, 512]),
    'lstm_recurrent_dropout': hp.uniform('lstm_recurrent_dropout', 0, 0.7),
    'lstm_dropout': hp.uniform('lstm_dropout', 0, 0.7),
    'highway_layer_count': hp.choice('highway_layer_count', [4, 8, 16, 32, 64, 128]),
    'highway_activation': hp.choice('highway_activation', ['relu', 'tanh']),
    'dropout': hp.uniform('dropout', 0, 0.7),
    'batch_size': hp.choice('batch_size', [16, 32, 64, 128, 256]),
    'optimizer': hp.choice('optimizer', ['rmsprop', 'adam', 'sgd']),
    'loss': hp.choice('loss', ['mean_square_error', 'mean_absolute_error'])
}

def objective(configuration):

    print(configuration)
    return 256 - configuration["batch_size"]

best = fmin(objective,
    space=space,
    algo=tpe.suggest,
    max_evals=10)

print("BEST:")
print(best)
