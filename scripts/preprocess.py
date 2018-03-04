import numpy as np
import spacy
import xml.etree.ElementTree as et
import codecs, json
import gzip
import pickle

nlp = spacy.load('en')

def getVectors(text, max_length, embedding_size):
    vectors = np.zeros((max_length, embedding_size))
    if text != None:
        doc = nlp(text)
        j = max(0, max_length - len(doc))  
        for token in doc:
            vectors[j] = np.array(token.vector)
            j=j+1
            if j==max_length:
                break
    return vectors


tree = et.parse('./full_consolidated.xml')
print("XML parsing finished")
root = tree.getroot()

skip_records = 0
process_records = 39031

i = 0
skipped = 0
max_summary_length = 30
max_description_length = 200

dataset_ids = np.zeros((process_records))
project_ids = np.zeros((process_records))
summary_vectors = np.zeros((process_records, max_summary_length, 384))
description_vectors = np.zeros((process_records, max_description_length, 384))
y = np.zeros((process_records))

for item in root:
    dataset_id = int(item.find('dataset_id').text)
    project_id = int(item.find('project_id').text)
    summary = item.find('summary').text
    description = item.find('description').text
    time_spent_in_seconds = int(item.find('seconds').text)

    if time_spent_in_seconds < 600 or time_spent_in_seconds > 101241:
        continue

    if skipped < skip_records:
        skipped = skipped + 1
        continue

    dataset_ids[i] = dataset_id
    project_ids[i] = project_id
    summary_vectors[i] = getVectors(summary, max_summary_length, 384)
    description_vectors[i] = getVectors(description, max_description_length, 384)
    y[i] = time_spent_in_seconds

    i=i+1
    print("Records processed:", i)
    if i >= process_records:
        break

with gzip.open("preprocessed-data/dataset-ids.pkl.gz", "wb") as file:
    pickle.dump(dataset_ids, file, 4)

with gzip.open("preprocessed-data/project-ids.pkl.gz", "wb") as file:
    pickle.dump(project_ids, file, 4)

with gzip.open("preprocessed-data/summary-vectors.pkl.gz", "wb") as file:
    pickle.dump(summary_vectors, file, 4)

with gzip.open("preprocessed-data/y.pkl.gz", "wb") as file:
    pickle.dump(y, file, 4)

parts = 10
part_size = process_records // parts
for i in range(parts):
    with gzip.open("preprocessed-data/description-vectors_%d.pkl.gz" % (i + 1), "wb") as file:
        pickle.dump(description_vectors[i * part_size: process_records if i == parts else (i + 1) * part_size], file, 4)


