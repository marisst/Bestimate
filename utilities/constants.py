CONFIGURATIONS_FOLDER = "configurations"
DATA_FOLDER = "raw_data"
DATASET_FOLDER = "training_datasets"
RESULTS_FOLDER = "results"
DATA_COLLECTION_FOLDER = "data_collection"

STATISTICS_FOLDER = DATASET_FOLDER + "/statistics"

URL_PREFIX = "https://"
JIRA_REST = "/rest/api/latest"
JIRA_SEARCH = "/search"

RAW_POSTFIX = "raw"
CLEANED_POSTFIX = "clean"
MERGED_POSTFIX = "merged"
FILTERED_POSTFIX = "filtered"
TOKEN_COUNT_POSTFIX = "tokens"
NUMERIC_POSTFIX = "numeric"
EMB_POSTFIX = "emb"
EMB2DIM_POSTFIX = "emb2dim"
VECTORIZED_POSTFIX = "vectorized"
SPACY_LOOKUP_POSTFIX = "spacy_lookup"
GENSIM_MODEL = "gensim_model"

POTENTIAL_REPOS_FILENAME = "%s/%s" % (DATA_COLLECTION_FOLDER, "potential_repos.txt")

JSON_FILE_EXTENSION = ".json"
CSV_FILE_EXTENSION = ".csv"
HDF5_FILE_EXTENSION = ".hdf5"
PICKLE_FILE_EXTENSION = ".pkl"
PNG_FILE_XTENSION = ".png"
TEXT_FILE_EXTENSION = ".txt"

LABELED_FILENAME = "lab"
UNLABELED_FILENAME = "unl"
ALL_FILENAME = "all"
RESULTS_FILENAME = "results"
CONFIGURATION_FILENAME = "configuration"

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

LABELED_DATA_JQL = "timespent > 0 and resolution != Unresolved"
UNLABELED_DATA_JQL = "timespent <= 0 or timespent is EMPTY or resolution is EMPTY"

DESCRIPTION_FIELD_KEY = "description"
SUMMARY_FIELD_KEY = "summary"
PROJECT_FIELD_KEY = "project"
TIMESPENT_FIELD_KEY = "timespent"
ID_FIELD_KEY = "id"
FIELD_KEYS = ID_FIELD_KEY, PROJECT_FIELD_KEY, SUMMARY_FIELD_KEY, DESCRIPTION_FIELD_KEY, TIMESPENT_FIELD_KEY

ALPHA_FIELD = "alpha"
TOTAL_KEY = "total"
NUMERIC_TEXT_KEY = "numeric_text"
PRELEARNING = "pre"

def get_repository_filename(repository_name, labeling, data_type, extension):
    return "%s/%s/%s_%s_%s%s" % (DATA_FOLDER, repository_name, repository_name, labeling, data_type, extension)

def get_dataset_filename(dataset_name, labeling, data_type, extension):
    return "%s/%s/%s_%s_%s%s" % (DATASET_FOLDER, dataset_name, dataset_name, labeling, data_type, extension)

def get_running_configuration_filename(configuration_name):
    return "%s/%s%s" % (CONFIGURATIONS_FOLDER, configuration_name, JSON_FILE_EXTENSION)

def get_repository_search_url(repository_base_url):
    return URL_PREFIX + repository_base_url + JIRA_REST + JIRA_SEARCH

def get_results_folder_name(dataset, training_session_name):
    return "%s/%s-%s" % (RESULTS_FOLDER, dataset, training_session_name)

def get_statistics_image_filename(dataset, stat_name):
    return "%s/%s-%s%s" % (STATISTICS_FOLDER, dataset, stat_name, PNG_FILE_XTENSION)

def get_results_filename(dataset, training_session_name):
    return get_results_folder_name(dataset, training_session_name) + "/weights-{epoch:04d}-{val_loss:.0f}" + HDF5_FILE_EXTENSION
