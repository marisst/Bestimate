FROM tensorflow/tensorflow:1.7.1-devel-gpu-py3

WORKDIR /app

ADD requirements.txt /app

RUN pip install -r requirements.txt
RUN python -m spacy download en_vectors_web_lg

ADD data_preprocessing /app/data_preprocessing
ADD embedding_pretraining /app/embedding_pretraining
ADD training /app/training
ADD utilities /app/utilities

ADD training_datasets/all/all_unl_merged.json /app/training_datasets/all/all_unl_merged.json
ADD training_datasets/all/all_lab_merged.json /app/training_datasets/all/all_lab_merged.json

CMD ["python", "-m", "training.hypopt", "all", "spacy"]