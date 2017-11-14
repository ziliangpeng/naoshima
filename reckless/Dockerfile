FROM python:2.7

MAINTAINER Victor Peng version: 0.1

RUN pip install web.py

ADD . reckless/
WORKDIR "/reckless"

ENTRYPOINT ["python", "run.py", "80"]
