import spacy
import sys

from utilities.constants import get_dataset_filename, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION, SPACY_LOOKUP_POSTFIX
from utilities.file_utils import load_json, save_json
from utilities.string_utils import get_part_strings


def spacy_lookup(dataset):

    dictionary_filename = get_dataset_filename(dataset, ALL_FILENAME, DICTIONARY_POSTFIX, JSON_FILE_EXTENSION)
    dictionary = load_json(dictionary_filename)

    nlp = spacy.load('en_vectors_web_lg')

    print("Creating lookup table...")
    no_vector_count = 0
    lookup = {}
    for word in dictionary:

        doc = nlp(word)
        if doc[0].has_vector == False:
            no_vector_count += 1
            continue

        lookup[word] = doc[0].vector.tolist()

    print("%d (%.0f%%) of %d dictionary words had vectors" % get_part_strings(len(lookup), len(lookup) + no_vector_count))

    print("Saving...")
    lookup_filename = get_dataset_filename(dataset, ALL_FILENAME, SPACY_LOOKUP_POSTFIX, JSON_FILE_EXTENSION)
    save_json(lookup_filename, lookup)
    print("Lookup table saved at", lookup_filename)