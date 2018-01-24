FROM ubuntu:12.04
MAINTAINER Ziliang Peng

RUN apt-get update
RUN apt-get -y install python

ADD net_common.py /usr/src/
ADD server_simple_threading.py /usr/src/

EXPOSE 5000

CMD ["python", "/usr/src/server_simple_threading.py"]

