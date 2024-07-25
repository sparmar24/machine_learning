# syntax=docker/dockerfile:1

FROM python:3.11.9-slim-bullseye

ARG WORK_HOME=/usr/local/machine-learning

WORKDIR ${WORK_HOME}
COPY pyproject.toml .
COPY data/Breast_Cancer_Data.csv data/.
COPY src/modelvalidation_casestudy.py src/. 

RUN apt update \
  && apt install -y vim \
  && pip install uv \
  && uv venv \
  && uv pip compile pyproject.toml -o requirements.txt \
  && uv pip install -r requirements.txt 
