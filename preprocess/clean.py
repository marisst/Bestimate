import csv
import json
import numpy as np
import os
import sys
from string import punctuation
import re

from utilities import load_data, input_parser
from utilities.constants import *

NO_TEXT_TAGS = "code", "noformat"
ESCAPE_TAGS = "color", "quote", "anchor", "panel"
ESCAPE_STRINGS = "\\r", "\\n", "\\t", "\\f", "\\v", "\"", "\\\\", "h1. ", "h2. ", "h3. ", "h4. ", "h5. ", "h6. "
LINK_STARTERS = r"\#", r"\^", r"http\:\/\/", r"https\:\/\/", r"malto\:", r"file\:", r"\~"

def escape_tags_and_content(text, tags):

    for tag in tags:
        regex_matching_tag = re.compile("\{%s(.*?)\}(.*?)\{%s\}" % (tag, tag), re.DOTALL)
        text = re.sub(regex_matching_tag, "", text)

    return text

def escape_tags(text, tags):

    for tag in tags:
        text = re.sub("\{%s(.*?)\}" % tag, "", text)

    return text

def escape_strings(text, escape_strings):

    for escape_string in escape_strings:
        text = text.replace(escape_string, " ")

    return text

def escape_links(text, link_starters):

    for link_starter in link_starters:
        text = re.sub("\[(.*?\\|)?%s(.*?)\]" % link_starter, "", text)
        text = re.sub(r"\bhttps?://\S+", "", text)

    return text

def escape_stack_trace(text):

    text = re.sub(r"(at(\s+(\S+\s+){1,2}?)){3,}", "", text)

    return text

def escape_hex_character_codes(text):

    return re.sub(r"\\x\w\w", "", text)

def escape_punctuation_boundaries(text):

    return " ".join([word.lstrip(punctuation) for word in text.split()])

def escape_low_alpha_density_words(text):

    return " ".join([word for word in text.split() if calculate_alpha_density(word) > 0.93])

def remove_repeating_fragments(text):

    MAX_FRAGMENT_LENGTH = 15

    words = text.split()
    duplicates = np.full((len(words)), False)
    for i in range(1, len(words)):
        max_step = min(MAX_FRAGMENT_LENGTH, (i + 1) // 2)
        for step in range(1, max_step + 1):
            first_fragment = ' '.join(words[i-step+1:i+1])
            second_fragment = ' '.join(words[i-2*step+1:i-step+1])
            if first_fragment == second_fragment:
                duplicates[i-step+1:i+1] = True

    result = []
    for i, word in enumerate(words):
        if not duplicates[i]:
            result.append(word)

    return " ".join(result)

def escape_odd_spaces(text):
    
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text

def calculate_alpha_density(text):

    total = len(text)
    alphas = len(re.findall("[a-zA-Z]", text))
    spaces = len(re.findall("\s", text))
    symnums = total - (spaces + alphas)

    apos = text.count(r"\\'") * 3
    alphas = alphas + apos
    symnums = symnums - apos
    
    return alphas / (symnums + alphas) if (symnums + alphas) > 0 else 0

def clean(text):

    text = escape_tags_and_content(text, NO_TEXT_TAGS)
    text = escape_tags(text, ESCAPE_TAGS)
    text = escape_strings(text, ESCAPE_STRINGS)
    text = escape_links(text, LINK_STARTERS)
    text = escape_stack_trace(text)
    text = escape_hex_character_codes(text)
    text = escape_punctuation_boundaries(text)
    text = escape_low_alpha_density_words(text)
    text = remove_repeating_fragments(text)
    text = escape_odd_spaces(text)
    text = text.lower()
    return text

def load_file(filename):

    if not os.path.isfile(filename):
        print("File %s does not exist" % filename)
        return

    data = load_data.load_csv(filename, FIELD_KEYS)

    if data is None:
        print("Skipping cleaning %s because it does not consist any data" % filename)
        return

    return data

def get_clean_content(filename):

    data = load_file(filename)
    if data is None:
        return

    print("Cleaning %s" % filename)
    for i, datapoint in enumerate(data):
        
        if SUMMARY_FIELD_KEY in datapoint:
            datapoint[SUMMARY_FIELD_KEY] = clean(datapoint[SUMMARY_FIELD_KEY])
        
        if DESCRIPTION_FIELD_KEY in datapoint:
            
            clean_description = clean(datapoint[DESCRIPTION_FIELD_KEY])
            
            if clean_description != None and clean_description != "":
                datapoint[DESCRIPTION_FIELD_KEY] = clean_description
                datapoint[ALPHA_FIELD] = int("%.0f" % (calculate_alpha_density(datapoint[DESCRIPTION_FIELD_KEY]) * 100))
            else:
                datapoint.pop(DESCRIPTION_FIELD_KEY, None)

        if (i + 1) % 1000 == 0 or (i + 1) == len(data):
            percentage = (i + 1) / len(data) * 100
            print("%d (%.2f%%) of %d records cleaned" % (i + 1, percentage, len(data)))

    return sorted(data, key = lambda datapoint: datapoint[ALPHA_FIELD] if ALPHA_FIELD in datapoint else 101)

def clean_text(datasets_from_input):

    datasets = input_parser.select_datasets(datasets_from_input)
    
    if len(datasets) > 0:
        print("Cleaning text in the following dataset%s:" % ("s" if len(datasets) > 1 else ""), ", ".join(datasets))
    else:
        print("No datasets selected")
        return        

    for dataset_name in datasets:

        labeled_data_filename = get_dataset_filename(dataset_name, LABELED_FILENAME, RAW_POSTFIX, CSV_FILE_EXTENSION)
        labeled_cleaned_data_filename = get_dataset_filename(dataset_name, LABELED_FILENAME, CLEANED_POSTFIX, JSON_FILE_EXTENSION)
        clean_labeled_content = get_clean_content(labeled_data_filename)
        load_data.save_json(labeled_cleaned_data_filename, clean_labeled_content)
        print("Cleaned data saved at", labeled_cleaned_data_filename)

        unlabeled_data_filename = get_dataset_filename(dataset_name, UNLABELED_FILENAME, RAW_POSTFIX, CSV_FILE_EXTENSION)
        unlabeled_cleaned_data_filename = get_dataset_filename(dataset_name, UNLABELED_FILENAME, CLEANED_POSTFIX, JSON_FILE_EXTENSION)
        clean_unlabeled_content = get_clean_content(unlabeled_data_filename)
        load_data.save_json(unlabeled_cleaned_data_filename, clean_unlabeled_content)
        print("Cleaned data saved at", unlabeled_cleaned_data_filename)

datasets_from_input = sys.argv[1:]
clean_text(datasets_from_input)
