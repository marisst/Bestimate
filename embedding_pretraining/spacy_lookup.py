import spacy
import sys

from utilities.constants import get_dataset_filename, ALL_FILENAME, TOKEN_COUNT_POSTFIX, JSON_FILE_EXTENSION, SPACY_LOOKUP_POSTFIX
from utilities.file_utils import load_json, save_json
from utilities.string_utils import get_part_strings


def spacy_lookup(dataset, notes_filename, token_counts=None, save=True):

    if token_counts is None:
        token_count_filename = get_dataset_filename(dataset, ALL_FILENAME, TOKEN_COUNT_POSTFIX, JSON_FILE_EXTENSION)
        token_counts = load_json(token_count_filename)

    nlp = spacy.load('en_vectors_web_lg')

    print("Creating lookup table...")
    no_vector_count = 0
    lookup = {}
    for word in token_counts:

        doc = nlp(word[0])
        if doc[0].has_vector == False:
            no_vector_count += 1
            continue

        lookup[word[0]] = doc[0].vector.tolist()

    with open(notes_filename, "a") as notes_file:
        print("%d (%.0f%%) of %d dictionary words had Spacy vectors" % get_part_strings(len(lookup), len(lookup) + no_vector_count), file=notes_file)

    if save == True:
        print("Saving...")
        lookup_filename = get_dataset_filename(dataset, ALL_FILENAME, SPACY_LOOKUP_POSTFIX, JSON_FILE_EXTENSION)
        save_json(lookup_filename, lookup)
        print("Lookup table saved at", lookup_filename)

    return lookup