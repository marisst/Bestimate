CONFIGURATIONS_FOLDER = "configurations"
DATA_FOLDER = "data"
WEIGTHS_FOLDER = "weigths"

MERGED_DATA_FOLDER = DATA_FOLDER + "/merged"
FILTERED_DATA_FOLDER = DATA_FOLDER + "/filtered"
TOKEN_COUNT_DATA_FOLDER = DATA_FOLDER + "/token_counts"
STATISTICS_FOLDER = DATA_FOLDER + "/statistics"

URL_PREFIX = "https://"
JIRA_REST = "/rest/api/latest"
JIRA_SEARCH = "/search"

RAW_POSTFIX = "raw"
CLEANED_POSTFIX = "cln"

JSON_FILE_EXTENSION = ".json"
CSV_FILE_EXTENSION = ".csv"
HDF5_FILE_EXTENSION = ".hdf5"
PICKLE_FILE_EXTENSION = ".pkl"
PNG_FILE_XTENSION = ".png"
TEXT_FILE_EXTENSION = ".txt"

LABELED_FILENAME = "lab"
UNLABELED_FILENAME = "unl"
RESULTS_FILENAME = "results"
CONFIGURATION_FILENAME = "configuration"
REPO_LIST_FILENAME = "open_repositories"

LABEL_DISTRIBUTION_STAT = "label_distribution"
PROJECT_SIZE_STAT = "project_size"
TEXT_LENGTH_STAT = "text_length"

JSON_INDENT = 4
PICKLE_PROTOCOL = 4
SECONDS_IN_MINUTE = 60
SECONDS_IN_HOUR = 3600
SPACY_EMBEDDING_SIZE = 384
REQUEST_TIMEOUT_SECONDS = 15

PLOT_BBOX_INCHES = "tight"
OSX_PLATFORM_SYSTEM = "Darwin"

BING = "bing"
GOOGLE = "google"
SEARCH_ENGINES = [BING, GOOGLE]

LABELED_DATA_JQL = "timespent > 0 and resolution != Unresolved"
UNLABELED_DATA_JQL = "timespent <= 0 or timespent is EMPTY or resolution is EMPTY"

DESCRIPTION_FIELD_KEY = "description"
SUMMARY_FIELD_KEY = "summary"
PROJECT_FIELD_KEY = "project"
TIMESPENT_FIELD_KEY = "timespent"
FIELD_KEYS = PROJECT_FIELD_KEY, SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TIMESPENT_FIELD_KEY

ALPHA_FIELD = "alpha"

def get_running_configuration_filename(configuration_name):
    return "%s/%s%s" % (CONFIGURATIONS_FOLDER, configuration_name, JSON_FILE_EXTENSION)

def get_repository_search_url(repository_base_url):
    return URL_PREFIX + repository_base_url + JIRA_REST + JIRA_SEARCH

def get_repo_list_filename(search_engine):
    return "%s/%s_%s%s" % (DATA_FOLDER, REPO_LIST_FILENAME, search_engine, JSON_FILE_EXTENSION)

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

def get_filtered_dataset_filename(dataset_name):
    return "%s/%s%s" % (FILTERED_DATA_FOLDER, dataset_name, JSON_FILE_EXTENSION)

def get_token_count_filename(dataset_name):
    return "%s/%s%s" % (TOKEN_COUNT_DATA_FOLDER, dataset_name, JSON_FILE_EXTENSION)

def get_weigths_folder_name(dataset, training_session_name):
    return "%s/%s-%s" % (WEIGTHS_FOLDER, dataset, training_session_name)

def get_statistics_image_filename(dataset, stat_name):
    return "%s/%s-%s%s" % (STATISTICS_FOLDER, dataset, stat_name, PNG_FILE_XTENSION)

def get_weigths_filename(dataset, training_session_name):
    return get_weigths_folder_name(dataset, training_session_name) + "/weights-{epoch:04d}-{val_loss:.0f}" + HDF5_FILE_EXTENSION

def get_results_filename(dataset, training_session_name):
    return "%s/%s%s" % (get_weigths_folder_name(dataset, training_session_name), RESULTS_FILENAME, CSV_FILE_EXTENSION)

def get_results_plot_filename(dataset, training_session_name):
    return "%s/%s%s" % (get_weigths_folder_name(dataset, training_session_name), RESULTS_FILENAME, PNG_FILE_XTENSION)

def get_configuration_filename(dataset, training_session_name):
    return "%s/%s%s" % (get_weigths_folder_name(dataset, training_session_name), CONFIGURATION_FILENAME, TEXT_FILE_EXTENSION)

def get_prediction_plot_filename(dataset, training_session_name):
    return get_weigths_folder_name(dataset, training_session_name) + "/weights-{epoch:04d}-{val_loss:.0f}-predictions" + PNG_FILE_XTENSION
