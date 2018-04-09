DATA_FOLDER = "data"
MERGED_DATA_FOLDER = DATA_FOLDER + "/merged"

URL_PREFIX = "https://"
JIRA_REST = "/rest/api/latest"
JIRA_SEARCH = "/search"

RAW_POSTFIX = "raw"
CLEANED_POSTFIX = "cln"

JSON_FILE_EXTENSION = ".json"
CSV_FILE_EXTENSION = ".csv"

LABELED_FILENAME = "lab"
UNLABELED_FILENAME = "unl"

JSON_INDENT = 4

DESCRIPTION_FIELD_KEY = "description"
SUMMARY_FIELD_KEY = "summary"
PROJECT_FIELD_KEY = "project"
TIMESPENT_FIELD_KEY = "timespent"
FIELD_KEYS = PROJECT_FIELD_KEY, SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TIMESPENT_FIELD_KEY

def get_folder_name(dataset_name):
    return "%s/%s" % (DATA_FOLDER, dataset_name)

def get_labeled_raw_filename(dataset_name):
    folder = get_folder_name(dataset_name)
    return "%s/%s_%s_%s%s" % (folder, dataset_name, LABELED_FILENAME, RAW_POSTFIX, CSV_FILE_EXTENSION)

def get_unlabeled_raw_filename(dataset_name):
    folder = get_folder_name(dataset_name)
    return "%s/%s_%s_%s%s" % (folder, dataset_name, UNLABELED_FILENAME, RAW_POSTFIX, CSV_FILE_EXTENSION)

def get_labeled_cleaned_filename(dataset_name):
    folder = get_folder_name(dataset_name)
    return "%s/%s_%s_%s%s" % (folder, dataset_name, LABELED_FILENAME, CLEANED_POSTFIX, JSON_FILE_EXTENSION)

def get_unlabeled_cleaned_filename(dataset_name):
    folder = get_folder_name(dataset_name)
    return "%s/%s_%s_%s%s" % (folder, dataset_name, UNLABELED_FILENAME, CLEANED_POSTFIX, JSON_FILE_EXTENSION)

def get_merged_dataset_filename(dataset_name):
    return "%s/%s%s" % (MERGED_DATA_FOLDER, dataset_name, JSON_FILE_EXTENSION)