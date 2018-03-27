DATA_FOLDER = "data"
URL_PREFIX = "https://"
JIRA_REST = "/rest/api/latest"
JIRA_SEARCH = "/search"
RAW_POSTFIX = "raw"
LABELED_DATA_FOLDER = "labeled"
UNLABELED_DATA_FOLDER = "unlabeled"
XML_ITEM_NAME = "item"
XML_ROOT_NAME = "items"
DATA_FILE_EXTENSION = ".json"
LABELED_FILENAME = "lab"
UNLABELED_FILENAME = "unl"

def get_folder_name(dataset_name):
    return "%s/%s" % (DATA_FOLDER, dataset_name)

def get_labeled_raw_filename(dataset_name):
    folder = get_folder_name(dataset_name)
    return "%s/%s_%s_%s%s" % (folder, dataset_name, LABELED_FILENAME, RAW_POSTFIX, DATA_FILE_EXTENSION)

def get_unlabeled_raw_filename(dataset_name):
    folder = get_folder_name(dataset_name)
    return "%s/%s_%s_%s%s" % (folder, dataset_name, UNLABELED_FILENAME, RAW_POSTFIX, DATA_FILE_EXTENSION)