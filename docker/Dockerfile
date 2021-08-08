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

RUN wget https://storage.googleapis.com/base_dados/models/model_2_best.h5

RUN mv model_2_best.h5 recognition/model.h5

EXPOSE 8000

# ENTRYPOINT [ "uvicorn", "main:app", "--host", "0.0.0.0", "--reload" ]
ENTRYPOINT [ "python", "main.py"]
