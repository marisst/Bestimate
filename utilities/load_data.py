import csv
import os
import json
import pickle
import sys

def load_json(filename):

    if not os.path.isfile(filename):
        print("File %s does not exist" % filename)
        return

    print("Parsing data from %s" % filename)
    
    return json.load(open(filename))

def load_csv(filename, keys):

    if not os.path.isfile(filename):
        print("File %s does not exist" % filename)
        return

    data = []

    csv.field_size_limit(2147483647)

    print("Parsing data from %s" % filename)

    with open(filename, 'r') as file:
        csvreader = csv.reader(file)
        for k, row in enumerate(csvreader):

            row_as_dict = {}
            for i, value in enumerate(row):
                if value is None or value == "":
                    continue
                row_as_dict[keys[i]] = value
            data.append(row_as_dict)
    
    return data

def load_pickle(filename):

    if not os.path.isfile(filename):
        print("File %s does not exist" % filename)
        return

    with open(filename, "rb") as file:
        return pickle.load(file)

def create_folder_if_needed(folder_name):

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_next_dataset_name(folder_name):

    existing_dataset_count = sum(1 for f in os.listdir(folder_name))
    return str(hex(existing_dataset_count + 1))[2::].upper()