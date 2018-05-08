import csv
import os
import json
import pickle
import platform
import shutil
import sys

from utilities.constants import *

MAX_BYTES = 2 ** 31 - 1

def load_json(filename):

    if not os.path.isfile(filename):
        print("File %s does not exist" % filename)
        return

    print("Parsing data from %s" % filename)
    
    return json.load(open(filename))

def save_json(filename, data):

    with open(filename, "w") as file:
        json.dump(data, file, indent=JSON_INDENT)

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

    if (platform.system() == OSX_PLATFORM_SYSTEM):
        bytes_in = bytearray(0)
        input_size = os.path.getsize(filename)
        with open(filename, 'rb') as f_in:
            for _ in range(0, input_size, MAX_BYTES):
                bytes_in += f_in.read(MAX_BYTES)
        return pickle.loads(bytes_in)

    with open(filename, "rb") as file:
        return pickle.load(file)


def save_pickle(filename, data):

    # need to use chunks on OS X because of a bug
    # https://stackoverflow.com/questions/31468117/python-3-can-pickle-handle-byte-objects-larger-than-4gb
    if platform.system() == OSX_PLATFORM_SYSTEM:
        bytes_out = pickle.dumps(data, protocol=PICKLE_PROTOCOL)
        with open(filename, 'wb') as file:
            for idx in range(0, len(bytes_out), MAX_BYTES):
                file.write(bytes_out[idx:idx+MAX_BYTES])
        print("Data saved on %s in chunks" % filename)
        return

    with open(filename, "wb") as file:
        pickle.dump(data, file, PICKLE_PROTOCOL)
    print("Data saved on %s" % filename)

def create_folder_if_needed(folder_name):

    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def get_next_dataset_name(folder):

    existing_dataset_count = sum(1 for f in os.listdir(folder))
    return str(existing_dataset_count + 1)

def create_dataset_folder(dataset_name, folder):

    if not os.path.exists(folder):
        os.makedirs(folder)
    
    dataset_folder = "%s/%s" % (folder, dataset_name)
    if os.path.exists(dataset_folder):
        if input("%s already exists, do you want to remove it's contents? (y/n) " % dataset_folder) != "y":
            sys.exit()
        shutil.rmtree(dataset_folder)
    os.makedirs(dataset_folder)

    return dataset_folder