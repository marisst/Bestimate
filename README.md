# Bestimate
Automated software development task estimation

## Fetching data from JIRA repository
```
python -m fetching.fetch_data DATASET jira.repositoryname.com
```
Replace `DATASET` with a name for a folder in which the fetched data will be saved and `jira.repositoryname.com` with the address of the JIRA repository from which you want to fetch the data. The fetched data will be saved in `/data/DATASET` subfolder in CSV format. Labeled and unlabeled datapoints are stored seperately. A new request to JIRA REST service is made for each 50 record chunk due to JIRA's constraints until all records are loaded. You can sign in to the targeted JIRA repository to gain access to more data by username and [API token](https://confluence.atlassian.com/cloud/api-tokens-938839638.html). If the API token is not working, a less secure alternative is to create a new user account and use its password instead of the API key.

## Preprocessing data
The fetched datapoints are further processed by cleaning textual task descriptions, merging data from several repositories together and filtering in order to increase data homogeinity or simply compare results of different datapoitn subsets.

### Clean text
```
python -m preprocessing.clean_text DATASET
```
Replace `DATASET` with the name of the dataset which you are willing to clean. If no dataset is selected, all saved datasets are cleaned. A list of datasets can be cleaned as shown below.
```
python -m preprocessing.clean_text DATASET1 DATASET2 DATASET3
```
The text is processed according to [Atlassian formatting notation](https://jira.atlassian.com/secure/WikiRendererHelpAction.jspa?section=all) as follows:
- text contained in `NO_TEXT_TAGS` such as `{code}` and the tags themselves are removed;
- `ESCAPE_TAGS` such as `{color}` are removed while the content is preserved;
- `ESCAPE_STRINGS` such as new line symbols and tabulators are replaced with spaces;
- internal and external links staring with `LINK_STARTERS` such as `http://` and `file\:` are removed;
- if the word "at" is followed by one or two words and this pattern is repeated at least 3 consequent times there is a high chance that it is a stack trace fragment and therefore is removed;
- characters that are neither latin letters, numbers nor punctuation marks and have therefor been converted to their hexadecimal representation are removed;
- finally, several consecutive spaces between words are replaced with one space and leading and trailing spaces are removed from text fragments.
Datapoints with cleaned text are saved in `data/DATASET` forlder in JSON format. The description field often contains fragments which does not belong to natural language but is rather text from debugging or different technical codes. To detect such text fragments, an *alpha density* is calculated as follows:
```
alpha = count(alpha_characters) / (count(all_characters) - count(spacing_characters))
```
where `alpha_characters` contain only latin letters.