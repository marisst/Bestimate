from enum import Enum
import sys

from preprocess.filter_config import FilterConfig
from preprocess.filter_module import load_dataset, filter_data
from utilities.constants import *
from utilities.string_utils import merge_sentences

class Extreme(Enum):
    MINIMUM = min
    MAXIMUM = max

def print_extreme(dataset, extreme):

    print("Getting label range...")
    data = load_dataset(dataset, LABELED_FILENAME)
    data = [datapoint for datapoint in data if datapoint.get(TIMESPENT_FIELD_KEY) is not None]
    extreme_datapoint = extreme.value(data, key=lambda datapoint: datapoint[TIMESPENT_FIELD_KEY])
    print("The %s timespent is %.2f hours for the following issue: %s"
        % (
            extreme.name,
            extreme_datapoint[TIMESPENT_FIELD_KEY] / SECONDS_IN_HOUR,
            merge_sentences(extreme_datapoint[SUMMARY_FIELD_KEY])))

sys_argv_count = len(sys.argv)

if (sys_argv_count < 2):
    print("Please choose a dataset to filter")
    sys.exit()

if (sys_argv_count > 2):
    print("Please choose one argument containing the name of the dataset you want to filter")
    sys.exit()

dataset = sys.argv[1]
filter_config = FilterConfig()

if input("Would you like to remove tasks with short textual descriptions? (y/n) ") == "y":
    filter_config.min_word_count = int(input("Please enter the minimum text length (words): "))

print_extreme(dataset, Extreme.MINIMUM)
print_extreme(dataset, Extreme.MAXIMUM)
if input("Would you like to remove extreme outliers? (y/n) ") == "y":
    filter_config.min_timespent_minutes = int(input("Please enter the lower bound (integer) in minutes: "))
    filter_config.max_timespent_minutes = int(input("Please enter the upper bound (integer) in minutes: "))

if input("Would you like to increase homogeneity by removing small projects? (y/n) ") == "y":
    filter_config.min_project_size = int(input("Please enter the minimum number of labeled filtered datapoints in a project: "))
    
if input("Would you like to make even distribution by removing skewed data? (y/n) ") == "y":
    filter_config.even_distribution_bin_count = int(input("Please choose number of bins: "))

filter_data(sys.argv[1], filter_config)