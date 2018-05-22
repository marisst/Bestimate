from training.hypopt import optimize_model
from threading import Thread

training_dataset_names = ["all", "exo", "ecms-exo", "tdf", "gzl", "hsc", "tup-tdf", "ezp-ezz", "apc-carbondata"]
embedding_types = {
    ("gensim", "skip-gram"),
    ("gensim", "CBOW"),
    ("spacy", None)
}
optimizers = ["rmsprop", "adam", "sgd"]
highway_activations = ["relu", "tanh"]
min_project_size = [1, 20, 50, 200, 500]


def run():
    
    for dataset in training_dataset_names:
        for embedding_type in embedding_types:
            for optimizer in optimizers:
                for highway_activation in highway_activations:

                    if dataset == "all":
                        for min_size in min_project_size:
                            args = (dataset, embedding_type, optimizer, highway_activation, min_size)
                        else:
                            args = (dataset, embedding_type, optimizer, highway_activation)

                    thread = Thread(target=optimize_model, args=args)
                    thread.start()
                    thread.join()
                    print("Thread finished")

if __name__ == "__main__":
    run()