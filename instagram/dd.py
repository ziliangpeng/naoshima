import os
import datadog
from logs import logger

DD_HOST_KEY = 'DD_HOST'

if DD_HOST_KEY in os.environ:
    dd_host = os.getenv(DD_HOST_KEY)
    statsd = datadog.DogStatsd(host=dd_host)
    logger.info("Datadog host is %s", dd_host)
else:
    logger.info("User default Datadog host")
    statsd = datadog.statsd

logger.info("dd host is actually %s", statsd.host)
