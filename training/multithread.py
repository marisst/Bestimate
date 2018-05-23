import sys
import subprocess

from training.hypopt import optimize_model

training_dataset_names = ["all", "exo", "ecms-exo", "tdf", "gzl", "hsc", "tup-tdf", "ezp-ezz", "carbondata-apc"]
embedding_types = ["gensim", "spacy"]


def run():

    for dataset in training_dataset_names:
        for embedding_type in embedding_types:
            subprocess.Popen(["python", "-m", "training.hypopt", dataset, embedding_type])


if __name__ == "__main__":
    run()