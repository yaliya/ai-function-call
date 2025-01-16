FROM python:3.12

ENV PYTHONUNBUFFERED=0

RUN pip3 install torch

RUN pip3 install numpy pandas transformers sentencepiece datasets accelerate transformers[torch]

WORKDIR /var/www