DATA_FOLDER = "data"
URL_PREFIX = "https://"
JIRA_REST = "/rest/api/latest"
JIRA_SEARCH = "/search"
RAW_POSTFIX = "raw"
CLEANED_POSTFIX = "cln"
JSON_FILE_EXTENSION = ".json"
CSV_FILE_EXTENSION = ".csv"
LABELED_FILENAME = "lab"
UNLABELED_FILENAME = "unl"
DESCRIPTION_FIELD = "description"
SUMMARY_FIELD = "summary"
JSON_INDENT = 4
FIELD_KEYS = "project", "summary", "description", "timespent"

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