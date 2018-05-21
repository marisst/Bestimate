FROM python:latest

WORKDIR /app

ADD . /app

RUN pip install -r requirements.txt
RUN pip install networkx==1.11
RUN python -m spacy download en_vectors_web_lg

CMD ["python", "-m", "training.hypopt", "1"]