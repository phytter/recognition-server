FROM tensorflow/tensorflow
 

RUN apt-get update -y && \
    apt-get install -y python-pip python-dev

COPY ./requirements.txt /code/requirements.txt

WORKDIR /code/

RUN apt-get install ffmpeg libjpeg-dev libsm6 libxext6  -y

RUN pip install -I pillow

RUN pip install -r requirements.txt


ENTRYPOINT [ "python", "server.py" ]
