FROM python:latest

WORKDIR /app

RUN pip install -r requirements.txt
RUN python -m spacy download en_vectors_web_lg

ADD . /app

CMD ["python", "-m", "training.hypopt", "1"]