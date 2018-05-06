FROM ubuntu:18.04

MAINTAINER Victor Peng version: 0.1

# Mostly fork from https://hub.docker.com/r/lupino/ardb-server/~/dockerfile/

RUN apt-get update && \
    apt-get install -y git make gcc g++ automake autoconf libbz2-dev libz-dev wget

RUN git clone --branch v0.9.7 https://github.com/yinqiwen/ardb.git ardb

ARG engine=rocksdb

RUN cd ardb && \
    storage_engine=$engine make && \
    cp src/ardb-server /usr/bin && \
    cp ardb.conf /etc && \
    cd .. && \
    yes | rm -r ardb

RUN sed -e 's@home.*@home /var/lib/ardb@' \
        -e 's/loglevel.*/loglevel info/' \
        -e 's/redis-compatible-mode.*no/redis-compatible-mode     yes/' \
        -i /etc/ardb.conf

RUN echo 'trusted-ip *.*.*.*' >> /etc/ardb.conf

RUN mv /etc/ardb.conf /etc/reckless-ardb.conf

WORKDIR /var/lib/ardb

EXPOSE 16379
ENTRYPOINT /usr/bin/ardb-server /etc/reckless-ardb.conf
