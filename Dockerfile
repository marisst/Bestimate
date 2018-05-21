FROM python:latest

WORKDIR /app

ADD requirements.txt /app

RUN pip install -r requirements.txt
RUN python -m spacy download en_vectors_web_lg

ADD data_preprocessing /app/data_preprocessing
ADD embedding_pretraining /app/embedding_pretraining
ADD training /app/training
ADD utilities /app/utilities

ADD training_datasets/2/2_lab_merged.json /app/training_datasets/2/2_lab_merged.json
ADD training_datasets/2/2_unl_merged.json /app/training_datasets/2/2_unl_merged.json

CMD ["python", "-m", "training.hypopt", "2", "gensim"]