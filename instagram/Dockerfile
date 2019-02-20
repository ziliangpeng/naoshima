FROM python:3.6.4-slim

MAINTAINER Victor Peng version: 0.1

# Only copy the requirements.txt for pip install, so caching works better
ADD pybot/requirements.txt instagram/requirements.txt

RUN pip install -r instagram/requirements.txt

# required for processing image to post
# This line has problem. ignore it for now.
# RUN imageio_download_bin ffmpeg

ADD pybot instagram/

ADD storage_config.local.docker instagram/storage_config.local

WORKDIR "/instagram"

ENTRYPOINT ["python", "bot.py", "-u"]
