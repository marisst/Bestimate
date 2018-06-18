import sys
import numpy as np
np.set_printoptions(threshold=np.nan)

from training import load_data as load
from training.data_generator import DataGenerator
from training import calculate_baselines as bsl
from data_preprocessing.filter_config import FilterConfig
from data_preprocessing.filter_data import filter_data

split_percentages = 60, 20

def fake_lookup(word):
    return [1, 2, 3]

def calculate_diffs(training_dataset_id):

    min_proj_sizes = [1, 200, 500, 1000]
    min_text_lengths = [1, 10, 20]

    baselines = np.zeros((len(min_proj_sizes) * len(min_text_lengths), 2))

    k = 0
    for min_proj_size in min_proj_sizes:
        for min_text_length in min_text_lengths:

            filter_config = FilterConfig()
            filter_config.min_timespent_minutes = 10
            filter_config.max_timespent_minutes = 960
            filter_config.min_word_count = min_text_length
            filter_config.min_project_size = min_proj_size
            labeled_data, _ = filter_data(training_dataset_id, filter_config, save=False)

            data, vector_dictionary = load.load_and_arrange(
            training_dataset_id,
            split_percentages,
            False,
            (5,5),
            fake_lookup,
            labeled_data=labeled_data)
            _, y_train, _, _, x_valid, y_valid = data

            training_median = np.median(y_train)
            training_mean = np.mean(y_train)
            validation_generator = DataGenerator(x_valid, y_valid, 512, False, (5,5), vector_dictionary)

            selected_validation_y = np.zeros((validation_generator.__len__() * 512))
            for batch_i in range(validation_generator.__len__()):
                _, y_batch = validation_generator.__getitem__(batch_i)
                for i in range(512):
                    selected_validation_y[batch_i * 512 + i] = y_batch[i]

            train_train_loss = min([bsl.mean_absolute_error(y_train, training_median), bsl.mean_absolute_error(y_train, training_mean)])
            train_valid_loss = min([bsl.mean_absolute_error(selected_validation_y, training_median), bsl.mean_absolute_error(selected_validation_y, training_mean)])

            baselines[k, 0] = train_train_loss
            baselines[k, 1] = train_valid_loss

            print("train-train", train_train_loss)
            print("train-valid", train_valid_loss)

            k = k + 1

    print(baselines)
    

if __name__ == "__main__":
    calculate_diffs(sys.argv[1])
