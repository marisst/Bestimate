import csv
import json
import numpy as np
import os
from string import punctuation
import re

from utilities.constants import ALPHA_FIELD, CLEANED_POSTFIX, CSV_FILE_EXTENSION, DESCRIPTION_FIELD_KEY, FIELD_KEYS
from utilities.constants import JSON_FILE_EXTENSION, LABELED_FILENAME, RAW_POSTFIX, SUMMARY_FIELD_KEY, UNLABELED_FILENAME
from utilities.constants import get_data_filename
from utilities.input_parser import select_datasets
from utilities.file_utils import load_csv, save_json

MAX_CHARS_PROCESSED = 10000
MIN_ALPHA_DENSITY = 0.93


def escape_tags_and_content(text):
    """Escape tags and their content containing text, which is not written in natural language, such as code snippets"""

    NO_TEXT_TAGS = "code", "noformat"
    for tag in NO_TEXT_TAGS:
        regex_matching_tag = re.compile("\{%s(.*?)\}(.*?)\{%s\}" % (tag, tag), re.DOTALL)
        text = re.sub(regex_matching_tag, "", text)

    return text


def escape_tags(text):
    """Escape markup tags, but retain their content"""

    ESCAPE_TAGS = "color", "quote", "anchor", "panel"
    for tag in  ESCAPE_TAGS:
        text = re.sub("\{%s(.*?)\}" % tag, "", text)

    return text


def escape_strings(text):
    """Escape line breaks, tabulators, slashes and JIRA heading markup symbols"""

    ESCAPE_STRINGS = "\\r", "\\n", "\\t", "\\f", "\\v", "\"", "\\\\", "h1. ", "h2. ", "h3. ", "h4. ", "h5. ", "h6. "
    for escape_string in ESCAPE_STRINGS:
        text = text.replace(escape_string, " ")

    return text


def escape_links(text):
    """Escape external and internal links, recognized by JIRA markup or leading 'http://' or 'https://' """

    LINK_STARTERS = r"\#", r"\^", r"http\:\/\/", r"https\:\/\/", r"malto\:", r"file\:", r"\~"
    for link_starter in LINK_STARTERS:
        text = re.sub("\[(.*?\\|)?%s(.*?)\]" % link_starter, "", text)
        text = re.sub(r"\bhttps?://\S+", "", text)

    return text


def escape_stack_trace(text):
    """Escape stack trace fragments which contain one or two words seperated by a space
    and follwing by the keyword 'at', repeated at least three times"""

    text = re.sub(r"(at(\s+(\S+\s+){1,2}?)){3,}", "", text)

    return text


def escape_hex_character_codes(text):
    """Escape characters outside the latin alphabet which are converted to hex code representation"""

    return re.sub(r"\\x\w\w", "", text)


def escape_punctuation_boundaries(text):
    """Remove all punctuation marks from the beginning and end of words,
    except for trailing period at the end of words"""

    return " ".join([word.strip(punctuation.replace(".", "")).lstrip(".") for word in text.split()])


def escape_low_alpha_density_words(text):
    """Escape words with low alpha density, except for those containing one apostrophe or one period"""

    clean_words = []
    for word in text.split():

        alpha_density = calculate_alpha_density(word)
        if alpha_density < MIN_ALPHA_DENSITY:
            allowed_symbol_count = word.count("'") + word.count(".")
            if word.count("'") > 1 or word.count(".") > 1 or alpha_density != (len(word) - allowed_symbol_count) / len(word):
                continue

        clean_words.append(word)

    return " ".join(clean_words)


def remove_repeating_fragments(text):
    """Remove repeating string fragments up to 15 words"""

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
    """Replace several consequent spaces with one space
    and remove spaces from string start and end"""
    
    text = re.sub(r"\s+", " ", text)
    text = text.strip()

    return text


def calculate_alpha_density(text):
    """Calculate alpha density which is the number of characters from a-zA-Z and apostrophe
    divided by the total number of characters"""

    total = len(text)
    alphas = len(re.findall("[a-zA-Z]", text))
    spaces = len(re.findall("\s", text))
    symnums = total - (spaces + alphas)

    apos = text.count(r"\\'") * 3
    alphas = alphas + apos
    symnums = symnums - apos
    
    return alphas / (symnums + alphas) if (symnums + alphas) > 0 else 0


def clean(text):
    """Clean and separate text in sentences"""

    text = text[:MAX_CHARS_PROCESSED]
    text = escape_tags_and_content(text)
    text = escape_tags(text)
    text = escape_strings(text)
    text = escape_links(text)
    text = escape_stack_trace(text)
    text = escape_hex_character_codes(text)
    text = escape_punctuation_boundaries(text)
    text = escape_low_alpha_density_words(text)
    text = remove_repeating_fragments(text)
    text = escape_odd_spaces(text)
    text = text.lower()

    if len(text) == 0:
        return None

    SENTENCE_SEPARATOR = ". "
    return [sentence.strip(".") for sentence in text.split(SENTENCE_SEPARATOR)]


def load_file(filename):
    """Load datapoints from CSV file if it exists or contains any records"""

    if not os.path.isfile(filename):
        print("File %s does not exist" % filename)
        return

    data = load_csv(filename, FIELD_KEYS)

    if data is None:
        print("Skipping cleaning %s because it does not consist any data" % filename)
        return

    return data


def get_clean_content(filename):
    """Load data from a file, reduce noise in task textual descriptions, separate text in sentences,
    calculate alpha density for description field sentences and return datapoints sorted by alpha density"""

    data = load_file(filename)
    if data is None:
        return

    print("Cleaning %s" % filename)
    for i, datapoint in enumerate(data):
        
        if SUMMARY_FIELD_KEY in datapoint:
            datapoint[SUMMARY_FIELD_KEY] = clean(datapoint[SUMMARY_FIELD_KEY])
        
        if DESCRIPTION_FIELD_KEY in datapoint:
            clean_description = clean(datapoint[DESCRIPTION_FIELD_KEY])
            if clean_description != None and len(clean_description) != 0:
                datapoint[DESCRIPTION_FIELD_KEY] = clean_description
                alpha_density = np.average(np.array([calculate_alpha_density(sentence) for sentence in datapoint[DESCRIPTION_FIELD_KEY]]))
                datapoint[ALPHA_FIELD] = int("%.0f" % (alpha_density * 100))
            else:
                datapoint.pop(DESCRIPTION_FIELD_KEY, None)

        if (i + 1) % 1000 == 0 or (i + 1) == len(data):
            percentage = (i + 1) / len(data) * 100
            print("%d (%.2f%%) of %d records cleaned" % (i + 1, percentage, len(data)))

    return sorted(data, key = lambda datapoint: datapoint[ALPHA_FIELD] if ALPHA_FIELD in datapoint else 101)


def clean_text(datasets_from_input):
    """Reduce noise from labeled and unlabeled task descriptions
    
    Arguments:

    datasets_from_input -- a list of identifiers of repositories which are to be cleaned,
    leave blank to clean text in all downloaded repositories
    """

    datasets = select_datasets(datasets_from_input)
    if len(datasets) > 0:
        print("Cleaning text in the following dataset%s:" % ("s" if len(datasets) > 1 else ""), ", ".join(datasets))
    else:
        print("No datasets selected")
        return        

    for dataset_name in datasets:

        for labeling in [LABELED_FILENAME, UNLABELED_FILENAME]:
            data_filename = get_data_filename(dataset_name, labeling, RAW_POSTFIX, CSV_FILE_EXTENSION)
            cleaned_data_filename = get_data_filename(dataset_name, labeling, CLEANED_POSTFIX, JSON_FILE_EXTENSION)
            clean_data = get_clean_content(data_filename)
            if clean_data is None or len(clean_data) == 0:
                continue
            save_json(cleaned_data_filename, clean_data)
            print("Cleaned data saved at", cleaned_data_filename)


if __name__ == "__main__":

    repositories = input("List one or more repository identifiers which you want to clean or leave blank and press ENTER to clean all downloaded data: ")
    clean_text(repositories)