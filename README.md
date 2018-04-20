# Bestimate
Automated software development task estimation

## Fetching Data from JIRA Repository
```
python -m fetching.fetch_data DATASET jira.repositoryname.com
```
Replace `DATASET` with a name for a folder in which the fetched data will be saved and `jira.repositoryname.com` with the address of the JIRA repository from which you want to fetch the data. The fetched data will be saved in `/data/DATASET` subfolder in CSV format. Labeled and unlabeled datapoints are stored seperately. A new request to JIRA REST service is made for each 50 record chunk due to JIRA's constraints until all records are loaded. You can sign in to the targeted JIRA repository by username and [API token](https://confluence.atlassian.com/cloud/api-tokens-938839638.html) to gain access to more data. If the API token is not working, an alternative is to create a new user account and use its password instead of the API key.

## Preprocessing Data
The fetched datapoints are further processed by cleaning textual task descriptions, merging data from several repositories together and filtering in order to increase data homogeinity or simply compare results of different datapoitn subsets.

### Cleaning Text
```
python -m preprocessing.clean_text DATASET1 DATASET2 DATASET3
```
A single dataset or a list of datasets can get cleaned as shown above. If no datasets are selected, all available datasets will get cleaned. The text is processed according to [Atlassian formatting notation](https://jira.atlassian.com/secure/WikiRendererHelpAction.jspa?section=all) as follows:
- text contained in `NO_TEXT_TAGS` such as `{code}` and the tags themselves are removed;
- `ESCAPE_TAGS` such as `{color}` are removed while the content is preserved;
- `ESCAPE_STRINGS` such as new line symbols and tabulators are replaced with spaces;
- internal and external links staring with `LINK_STARTERS` such as `http://` and `file\:` are removed;
- if the word "at" is followed by one or two words and this pattern is repeated at least 3 consequent times there is a high chance that it is a stack trace fragment and therefore is removed;
- characters that are neither latin letters, numbers nor punctuation marks and have therefor been converted to their hexadecimal representation are removed;
- finally, several consecutive spaces between words are replaced with one space and leading and trailing spaces are removed from text fragments.

Datapoints with cleaned text are saved in `data/DATASET` forlder in JSON format. The description field often contains fragments which does not belong to natural language but is rather text from debugging or different technical codes. To detect such text fragments, an *alpha density* is calculated as follows:
```
alpha = count(alpha_characters) / count(all__non_space_characters)
```
where `alpha_characters` contain only latin letters.

### Selecting or Excluding Projects and Merging Datasets
Datasets for model training and testing are composed from the cleaned data fetched from JIRA repositories. At this stage data from several JIRA repositories can be merged together and particular projects can be selected or excluded from the training and testing datasets.
```
python -m preprocessing.merge_data DATASET1 DATASET2
```
If you whish to create a training and testing dataset from one repository only, just pass the name of that single repository. If no datasets are selected, all available datasets will get merged together in a new dataset. Each new merged dataset is automatically assigned a name which is their hexadecimal sequence number and saved in `data/merged` folder.

### Filtering Data
Datapoints with short textual descriptions, extreme outliers and small projects can be removed as well as skewed data distributions can be made even by using the filtering module.
```
python -m preprocessing.filter_data A
```
Replace `A` with the hexadecimal sequence number of the training and testing dataset which you wish to filter. Skewed data distributions are made even by removing datapoints from any bins that have larger population than the smallest one.