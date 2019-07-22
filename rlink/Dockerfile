FROM python:3.7.2

MAINTAINER Victor Peng version: 0.1

ADD requirements.txt /rlink/

RUN pip install -r /rlink/requirements.txt

ADD . /rlink/

WORKDIR "/rlink"

EXPOSE 80

ENTRYPOINT ["python", "main.py"]
