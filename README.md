# Bestimate
Automated software development task estimation

## Fetching Data from JIRA Repository
```
python -m fetching.fetch_data DATASET jira.repositoryname.com
```
Replace `DATASET` with a name for a folder in which the fetched data will be saved and `jira.repositoryname.com` with the address of the JIRA repository from which you want to fetch the data. The fetched data will be saved in `/data/DATASET` subfolder in CSV format. Labeled and unlabeled datapoints are stored seperately. A new request to JIRA REST service is made for each 50 record chunk due to JIRA's constraints until all records are loaded. You can sign in to the targeted JIRA repository by username and [API token](https://confluence.atlassian.com/cloud/api-tokens-938839638.html) to gain access to more data. If the API token is not working, an alternative is to create a new user account and use its password instead of the API key.

## Preprocessing Data
The fetched datapoints are further processed by cleaning textual task descriptions, merging data from several repositories together and filtering them in order to increase data homogeinity.

### Cleaning Text
```
python -m preprocessing.clean_text DATASET1 DATASET2 DATASET3
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
python -m preprocessing.merge_data DATASET1 DATASET2
```
If you whish to create a training and testing dataset from one repository only, just pass the name of that single repository. If no datasets are selected, all available datasets will get merged together in a new dataset. Each new merged dataset is automatically assigned a hexadecimal sequence number and saved in `data/merged` folder.

### Filtering Data
Datapoints with short textual descriptions, extreme outliers and small projects can be removed as well as skewed data distributions can be made even by using the filtering module. The module can also make skewed data distributions even by removing datapoints from any bins that are more populated than the least populated one.
```
python -m preprocessing.filter_data A
```
Replace `A` with the hexadecimal sequence number of the training and testing dataset which you wish to filter.

## Statistics
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

```

