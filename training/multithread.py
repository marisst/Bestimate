from training.hypopt import optimize_model
from threading import Thread

training_dataset_names = ["all", "exo", "ecms-exo", "tdf", "gzl", "hsc", "tup-tdf", "ezp-ezz"]
embedding_types = {
    ("gensim", "skip-gram"),
    ("gensim", "CBOW"),
    ("spacy", None)
}
optimizers = ["rmsprop", "adam", "sgd"]
highway_activations = ["relu", "tanh"]


def run():
    
    for dataset in training_dataset_names:
        for embedding_type in embedding_types:
            for optimizer in optimizers:
                for highway_activation in highway_activations:

                    thread = Thread(target=optimize_model, args=(dataset, embedding_type, optimizer, highway_activation))
                    thread.start()
                    thread.join()
                    print("Thread finished")

if __name__ == "__main__":
    run()