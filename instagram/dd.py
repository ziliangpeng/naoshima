import os
import datadog

import user_config_reader
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

    def __init__(self, sd_client, u):
        logger.info("dd host is actually %s", sd_client.host)
        self.sd = sd_client
        self.u = u

    def followed(self):
        self.sd.increment('naoshima.ig.follow', 1, tags=["user:" + self.u])

    def get_profile(self, success):
        self.sd.increment('naoshima.ig.get_profile', 1, tags=['user:' + self.u, 'success:' + str(success)])

    def ratelimit_exceeded(self):
        self.sd.increment('naoshima.ig.ratelimite_exceeded', 1, tags=['user:' + self.u])


u = user_config_reader.load_secrets()[0]
m = IGStatd(sd, u)
