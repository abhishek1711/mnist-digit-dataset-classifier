FROM jupyter/tensorflow-notebook

# RUN pip install joblib


COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt


RUN mkdir model
ENV MODEL_DIR=/home/jovyan/model
ENV MODEL_FILE=custom_model.h5
# ENV METADATA_FILE=metadata.json

COPY train.py ./train.py
COPY inference.py ./inference.py

RUN python3 train.py