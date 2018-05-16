# Bestimate
With this package you can train a neural network model to estimate JIRA issues from public and private repositories. The solution was developed as a part of a master thesis project at Norwegian University of Science and Technology in Spring 2018.

## 1. Data Collection
The model uses JIRA issue summary and description field text and reported time spent on issue completion to learn the relationship between them. If you will be running the model on a private JIRA repository, please jump over to [Fetching Data from JIRA Repository](#fetching-data-from-jira-repository). If you want to fetch data from several publicly available JIRA repositories, check [Bulk Fetch](#bulk-fetch). This package contains a list of publicy available JIRA repositories gathered by an exhaustive search on the Internet. However, you can use the commands described in the next section to find new publicly available repositories.

### Discovering Publicly Available JIRA Repositories
To find new repositories using [Bing Web Search API](https://azure.microsoft.com/en-us/services/cognitive-services/bing-web-search-api/) or [Google Custom Search Engine API](https://cse.google.com/cse/), run the following command:
```
python -m data_collection.discover_repos
```
You will be asked to provide Google API key and Google Custom Search Engine ID or Bing Web Search API key. Both search engines offer free trial versions of their products. Bing does not requre other setup than just a simple registration as opposed to Google which needs to be configured to search the whole web by following the first two steps in this [Stack Overflow answer](https://stackoverflow.com/a/37084643). Bing search results are not limited, but Google allows you to access only the first 100 search results for each query. Therefore you might want to collect the potential links manually, add them to [/data_collection/potential_repos.txt](data_collection/potential_repos.txt) file by separating each URL with a line break and then run the following command to test if they lead to a public JIRA repository:
```
python -m data_collection.test_repos
```

### Fetching Data from JIRA Repository
To fetch data  a single private or public repository, run the following command:
```
python -m data_collection.fetch_data
```
You will be asked to provide the URL of the JIRA repository, e.g. "jira.exoplatform.org" and an identifier of the repository, which will be used to identify the fetched data when composing a new training dataset. A new request to JIRA REST service is made for each 50 record chunk due to JIRA's constraints until all records are loaded. You can sign in to the JIRA repository by username and [API token](https://confluence.atlassian.com/cloud/api-tokens-938839638.html) to gain access to more data. If the API token is not working, an alternative is to create a new user account and use its password instead of the API key.

### Bulk Fetch

As a part of this research project, 33 publicly available JIRA repositories, each containing at least 100 resolved issues with time spent greater than zero, were found and listed at [/data_collection/known_repos.json](data_collection/known_repos.json). The total number of labeled issues exceeds 65,000 and the number of unlabeled issues exceeds 2,000,000. To fetch data from all of these these repositories by a single command, run:
```
python -m data_collection.bulk_fetch
```
You can add more repository identifier and URL pairs to [/data_collection/known_repos.json](data_collection/known_repos.json) to fetch those when you run the command. If a folder with a known repository identifier already exists in [/raw_data](raw_data) folder, it will not be reloaded when running the command unless you manually delete it.

## 2. Data Preprocessing
The fetched datapoints are further processed by cleaning textual task descriptions, merging data from several repositories together and filtering them in order to increase data homogeinity.

### Cleaning Text
```
python -m preprocess.clean DATASET1 DATASET2 DATASET3
```
A single dataset or a list of datasets can get cleaned as shown above. If no datasets are selected, all available datasets will get cleaned. The text is processed according to [Atlassian formatting notation](https://jira.atlassian.com/secure/WikiRendererHelpAction.jspa?section=all) as follows:
- text contained in tags such as `{code}` and `{noformat}` and the tags themselves are removed;
- tags such as `{color}` and `{quote}` are removed while the content is preserved;
- new line symbols and tabulators are replaced with spaces;
- internal and external links staring strings such as `http://` and `file\:` are removed;
- if the word "at" is followed by one or two words and this pattern is repeated at least 3 consequent times there is a high chance that it is a stack trace fragment and therefore is removed;
- characters that are neither latin letters, numbers nor punctuation marks and have therefor been converted to a hexadecimal representation are removed;
- finally, several consecutive spaces between words are replaced with one space and leading and trailing spaces are removed.

Datapoints with cleaned text are saved in `data/DATASET` forlder in JSON format. The `description` field often contains fragments which do not belong to natural language but is rather text from debugging or different technical codes. To recognize such text fragments, *alpha density* is calculated for `description` field and added to the JSON file. *Alpha density* is a ratio between latin letter character count and the count of all characters except spaces.

### Selecting or Excluding Projects and Merging Datasets
Datasets for model training and testing are composed from the cleaned data fetched from JIRA repositories. At this stage data from several JIRA repositories can be merged together and particular projects can be selected or excluded from the training and testing datasets.
```
python -m preprocess.merge DATASET1 DATASET2
```
If you whish to create a training and testing dataset from one repository only, just pass the name of that single repository. If no datasets are selected, all available datasets will get merged together in a new dataset. Each new merged dataset is automatically assigned a hexadecimal sequence number and saved in `data/merged` folder.

### Filtering Data
Datapoints with short textual descriptions, extreme outliers and small projects can be removed as well as skewed data distributions can be made even by using the filtering module. The module can also make skewed data distributions even by removing datapoints from any bins that are more populated than the least populated one.
```
python -m preprocess.filter A
```
Replace `A` with the hexadecimal sequence number of the training and testing dataset which you wish to filter.

## Insights
The statistics module allow you to get to know the data better.

### Time Spent Distribution
```
python -m statistics.label_distribution A
```
To create a histogram of `timespent` field distribution, replace `A` with the hexadecimal sequence number of the training and testing dataset which you wish explore. Since the model tries to predict `timespent` it is also called "label". A histogram like the one below will be generated and saved in `data/statistics` directory.
![label_distribution example](readme_images/label_distribution_example.png)

### Project Size Distribution
```
python -m statistics.project_size A
```
To create a histogram of project size distribution, replace `A` with the hexadecimal sequence number of the training and testing dataset which you wish to explore. Project size is defined as the number of labeled issues in a project. A histogram like the one below will be generated and saved in `data/statistics` directory.
![project_size example](readme_images/project_size_example.png)

### Text Length Distribution
```
python -m statistics.text_length A
```
To create a histogram of text length, replace `A` with the hexadecimal sequence number of the training and testing dataset which you wish to explore. Text length is measured as the number of words in summary and description fields of labeled tasks. A histogram like the one below will be generated and saved in `data/statistics` directory.
![text_length example](readme_images/text_length_example.png)
