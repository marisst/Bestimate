import sys

from training.hypopt import optimize_model
from threading import Thread

training_dataset_names = ["all", "exo", "ecms-exo", "tdf", "gzl", "hsc", "tup-tdf", "ezp-ezz", "carbondata-apc"]

def run(embedding_type):
    
    threads = [Thread(target=optimize_model, args=(dataset, embedding_type)) for dataset in training_dataset_names]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print("ALL THREADS FINISHED")

if __name__ == "__main__":
    run(sys.argv[1])