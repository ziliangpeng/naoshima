FROM python:2.7

MAINTAINER Victor Peng version: 0.1

ADD requirements.txt /reckless/

RUN pip install -r /reckless/requirements.txt

ADD . /reckless/

WORKDIR "/reckless"

EXPOSE 80

ENTRYPOINT ["python", "run.py", "80"]
