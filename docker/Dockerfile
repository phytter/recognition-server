FROM tensorflow/tensorflow


RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

RUN mkdir app

COPY ./docker/requirements.txt /app/requirements.txt

COPY ./src /app/

WORKDIR /app/

RUN apt-get update --fix-missing

RUN apt-get install ffmpeg libjpeg-dev libsm6 libxext6  -y

RUN pip install -I pillow

RUN pip install -r requirements.txt

RUN apt-get install wget

RUN pip install fastapi[all]

RUN wget https://storage.googleapis.com/base_dados/models/my_data_model2_normal_extractormaster.h5

RUN mv my_data_model2_normal_extractormaster.h5 recognition/model.h5

ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--reload" ]
