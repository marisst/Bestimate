FROM python:latest

WORKDIR /app

ADD requirements.txt /app

RUN pip install -r requirements.txt
RUN python -m spacy download en_vectors_web_lg

ADD data_preprocessing /app/data_preprocessing
ADD embedding_pretraining /app/embedding_pretraining
ADD training /app/training
ADD utilities /app/utilities
ADD training_datasets /app/training_datasets

CMD ["python", "-m", "training.multithread"]