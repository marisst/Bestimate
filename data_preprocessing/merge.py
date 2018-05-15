import sys
from preprocess.merge_module import merge_data

merge_data(sys.argv[1:], True)