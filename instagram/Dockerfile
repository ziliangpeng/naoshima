FROM python:3.6.4-slim

MAINTAINER Victor Peng version: 0.1

ADD . instagram/

WORKDIR "/instagram"

RUN cp storage_config.local.docker storage_config.local

RUN pip install -r requirements.txt

ENTRYPOINT ["python", "bot.py", "-u"]
