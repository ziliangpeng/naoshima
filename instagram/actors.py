import random
import time

import config_reader
import data
import user_utils
from logs import logger
from data_repo import d0
from threading import Thread

from filter import Filter

"""
This file contains the actors that act on cron of trigger fashion. This is to distinguish the difference between this
and runner.py (long running program).

Actors can be triggered at any time, and does smaller incremental job.
"""


class ActorBase(Thread):
    # If we run actor once every hour, bot can follow 1000 / 24 = 41.6 users.
    # There is no need to generate more.
    # TODO: potentially we can read this value from config, or calculate based on number of actors in parallel
    TOTAL_LIMIT = 50

    # Randomize each batch of generation. 512s (8.5 min) should be able to randomize everything in a batch.
    # Each batch estimate to run for a few minutes
    RANDOM_RANGE = 512

    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.bot = d0.bot
        self.uid = self.bot.user_id

    def run(self):
        try:
            self.act()
        except Exception as e:
            logger.error(type(e))
            logger.error(e)

    def compute_score(self):
        return int(time.time()) + random.randint(-self.RANDOM_RANGE, self.RANDOM_RANGE)


class GenFoFoActor(ActorBase):
    FO_PER_FO_LIMIT = ActorBase.TOTAL_LIMIT

    def __init__(self, u):
        ActorBase.__init__(self, u)

    def act(self):
        conditions = config_reader.redis_load_conditions(self.u)
        total_generated = 0
        for foer_id, foer_name in user_utils.get_all_followers_gen(self.bot, self.uid):
            for fofoer_id, fofoer_name in user_utils.get_all_followers_gen(
                    self.bot, foer_id, max=self.FO_PER_FO_LIMIT):
                if data.is_followed(self.u, fofoer_id):
                    logger.info("Already followed %s. Skip.", fofoer_name)
                elif not Filter(fofoer_name, conditions).apply():
                    logger.info('%s has not passed filter', fofoer_name)
                else:
                    logger.info('Steal follower %s, %s', fofoer_name, fofoer_id)
                    data.set_id_to_name(fofoer_id, fofoer_name)
                    score = self.compute_score()
                    data.add_user_to_follow(self.u, fofoer_id, score)
                    total_generated += 1
                    if total_generated >= self.TOTAL_LIMIT:
                        return


if __name__ == '__main__':
    import sys
    name = sys.argv[1]
    a = GenFoFoActor(name)
    a.start()
