import sys
from multiprocessing import Pool

from training.hypopt import optimize_model

training_dataset_names = ["all"] #, "exo", "ecms-exo", "tdf", "gzl", "hsc", "tup-tdf", "ezp-ezz", "carbondata-apc"]
embedding_types = ["gensim", "spacy"]


def run():

    parameters = []
    for dataset in training_dataset_names:
        for embedding_type in embedding_types:
            parameters.append((dataset, embedding_type))

    pool_count = len(parameters)
    print("Pool count:", pool_count)

    with Pool(processes=pool_count) as pool:
        pool.starmap(optimize_model, parameters)

if __name__ == "__main__":
    run()