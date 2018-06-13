FROM python:3.6.4-slim

MAINTAINER Victor Peng version: 0.1

# Only copy the requirements.txt for pip install, so caching works better
ADD requirements.txt instagram/requirements.txt

RUN pip install -r instagram/requirements.txt

# required for processing image to post
RUN imageio_download_bin ffmpeg

ADD . instagram/

WORKDIR "/instagram"

RUN cp storage_config.local.docker storage_config.local

ENTRYPOINT ["python", "bot.py", "-u"]
