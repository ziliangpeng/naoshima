FROM ubuntu:18.04

MAINTAINER Victor Peng version: 0.1

# Mostly fork from https://hub.docker.com/r/lupino/ardb-server/~/dockerfile/

RUN apt-get update && \
    apt-get install -y git make gcc g++ automake autoconf libbz2-dev libz-dev wget

# TODO: if v0.9.6 is released, pin to it
RUN git clone https://github.com/yinqiwen/ardb.git ardb && \
    cd ardb && \
    storage_engine=rocksdb make && \
    cp src/ardb-server /usr/bin && \
    cp ardb.conf /etc && \
    cd .. && \
    yes | rm -r ardb

RUN sed -e 's@home.*@home /var/lib/ardb@' \
        -e 's/loglevel.*/loglevel info/' -i /etc/ardb.conf

RUN echo 'trusted-ip *.*.*.*' >> /etc/ardb.conf

RUN mv /etc/ardb.conf /etc/reckless-ardb.conf

WORKDIR /var/lib/ardb

EXPOSE 16379
ENTRYPOINT /usr/bin/ardb-server /etc/reckless-ardb.conf