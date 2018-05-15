import sys

from translate.dictionary_module import create_dictionary
from utilities.constants import *

if len(sys.argv) != 3:
    print("Please select one dataset and one of the following field keys:", SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TOTAL_KEY)
    sys.exit();

minimum_repetitions = int(input("Please enter minimum repetitions of a word to include in the dictionary: "))
create_dictionary(sys.argv[1], sys.argv[2], minimum_repetitions)