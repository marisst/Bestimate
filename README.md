# Bestimate
Automated software development task estimation

## Fetching data from JIRA repository
```
python fetch_data.py dataset_name jira.repositoryname.com
```
Replace `dataset_name` with a name for a folder in which the fetched data will be saved and `jira.repositoryname.com` with the address of the JIRA repository from which you want to fetch the data. The fetched data will be saved in `/data/folder_name` subfolder in JSON format. Labeled and unlabeled datapoints are stored seperately. A new request to JIRA REST service is made for each 50 record chunk due to JIRA's constraints until all records are loaded. You can sign in to the targeted JIRA repository to gain access to more data by username and [API token](https://confluence.atlassian.com/cloud/api-tokens-938839638.html). Data is fetched through JIRA REST interface's [search](https://developer.atlassian.com/cloud/jira/platform/rest/#api-api-2-search-get) endpoint.