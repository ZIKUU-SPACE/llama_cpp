FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV DEBCONF_NOWARNINGS=yes

RUN apt-get update && apt-get upgrade -y
RUN apt-get dist-upgrade -y && apt-get autoremove -y
RUN apt-get install -y build-essential wget
RUN apt-get install -y python3 python3-venv

WORKDIR /app

RUN wget -P ./models https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q6_K.gguf

COPY ./requirements.txt /app
COPY ./web.py /app

RUN python3 -m venv /llama_cpp
RUN . /llama_cpp/bin/activate && python -m pip install --upgrade pip
RUN . /llama_cpp/bin/activate && python -m pip install -r requirements.txt


COPY ./docker/entrypoint.sh /
RUN chmod 755 /entrypoint.sh

EXPOSE 8501
ENTRYPOINT ["/entrypoint.sh"]
