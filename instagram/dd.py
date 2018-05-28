import os
import datadog
from retrying import retry
from time import sleep
from collections import defaultdict

import user_config_reader
from logs import logger
import prometheus_client as pc
from threading import Thread

DD_HOST_KEY = 'DD_HOST'

if DD_HOST_KEY in os.environ:
    dd_host = os.getenv(DD_HOST_KEY)
    sd = datadog.DogStatsd(host=dd_host)
    logger.info("Datadog host is %s", dd_host)
else:
    logger.info("User default Datadog host")
    sd = datadog.statsd


PROM_HOST_KEY = 'PROM_HOST'
if PROM_HOST_KEY in os.environ:
    prom_host = "%s:9091" % (os.getenv(PROM_HOST_KEY))
    logger.info("Prometheus host is %s", prom_host)
else:
    prom_host = None
    logger.info("No prometheus host is given")

class PromPush(Thread):
    @retry
    def run(self):
        while (not sleep(5)):
            logger.info("Sending metrics to prometheus")
            pc.push_to_gateway(prom_host, job='ig-bot', registry=pc.REGISTRY)

if prom_host:
    PromPush().start()


class IGStatd:

    def __init__(self, sd_client, u):
        logger.info("dd host is actually %s", sd_client.host)
        self.sd = sd_client
        self.u = u
        self.prom_counters = defaultdict(pc.Counter)

    def followed(self):
        self.sd.increment('naoshima.ig.follow', 1, tags=["user:" + self.u])

    def get_profile(self, success):
        self.sd.increment('naoshima.ig.get_profile', 1, tags=['user:' + self.u, 'success:' + str(success)])
        self._prom_counter('naoshima:ig:get_profile', "Number of profile get", ['user']).labels(user=self.u).inc()

    def ratelimit_exceeded(self):
        self.sd.increment('naoshima.ig.ratelimite_exceeded', 1, tags=['user:' + self.u])

    def _prom_counter(self, name, description, labelnames=[]):
        if name in self.prom_counters:
            return self.prom_counters[name]
        else:
            try:
                c = pc.Counter(name, description, labelnames=labelnames)
                self.prom_counters[name] = c
                return c
            except ValueError:
                # magically handle race condition :-)
                return self.prom_counters[name]


u = user_config_reader.load_secrets()[0]
m = IGStatd(sd, u)





