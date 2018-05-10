import sys
from prima_model.vectorize_module import vectorize_dataset

max_text_length = int(input("Please input maximum text length: "))
vectorize_dataset(sys.argv[1], max_text_length)