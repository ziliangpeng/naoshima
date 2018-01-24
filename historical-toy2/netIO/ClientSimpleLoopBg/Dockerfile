FROM ubuntu:12.04
MAINTAINER Ziliang Peng

ADD net_common.py /usr/src/
ADD onetime_client.py /usr/src/
ADD onetime_client_loop_bg.sh /usr/src/

WORKDIR /usr/src

CMD ["./onetime_client_loop_bg.sh"]

