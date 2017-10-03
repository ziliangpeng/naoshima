FROM python:2.7

MAINTAINER Victor Peng version: 0.1

ADD . reckless/
WORKDIR "/reckless"

RUN pip install web.py
ENTRYPOINT ["python", "run.py", "80"]
