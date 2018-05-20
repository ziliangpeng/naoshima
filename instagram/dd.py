import os
import datadog
from logs import logger

DD_HOST_KEY = 'DD_HOST'

if DD_HOST_KEY in os.environ:
    dd_host = os.getenv(DD_HOST_KEY)
    sd = datadog.DogStatsd(host=dd_host)
    logger.info("Datadog host is %s", dd_host)
else:
    logger.info("User default Datadog host")
    sd = datadog.statsd


class IGStatd:

    def __init__(self, sd_client):
        logger.info("dd host is actually %s", sd_client.host)
        self.sd = sd_client


    def followed(self, user):
        self.sd.increment('naoshima.ig.follow', 1, tags=["user:" + user])


m = IGStatd(sd)
