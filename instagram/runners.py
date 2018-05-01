import datetime

import itertools

import logs
import random
import time
from threading import Thread
from retrying import retry
import traceback

import data
import user_config_reader
import user_utils
from filter import Filter
from queue import Queue
from data_repo import d0
from datadog import statsd

WHITELIST_USER = user_config_reader.load_whitelist()

logger = logs.logger

# TODO:
# 1. follow followers' followers
# 2. keep following top brand's latest followers
# 3. keep following hashtag's most recent users


class GenUnfo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.user_id = user_utils.get_user_id(u)  # TODO: check null
        self.bot = d0.bot
        self.queue_to_unfo = d0.queue_to_unfo

    @retry
    def run(self):
        r = random.Random()
        while True:
            try:
                follows = user_utils.get_follows(self.bot, self.user_id)
                # followers = utils.get_followers(self.bot, self.user_id)
                # self.id_name_dict.update(follows)
                # self.id_name_dict.update(followers)
                if len(follows) < user_config_reader.load_min_follow():
                    logger.info("Only %d follows. Pause.", len(follows))
                    time.sleep(60 * 30)
                    continue

                n = 100
                filtered_user_ids = [x for x in follows if data.get_id_to_name(x) not in WHITELIST_USER]
                to_unfo = random.sample(filtered_user_ids, n)
                random.shuffle(to_unfo)
                for i, f in enumerate(to_unfo):
                    logger.info('#%03d gen unfollow: %s', i, data.get_id_to_name(f))
                    if r.random() < 0.5:
                        self.queue_to_unfo.put(f)
                time.sleep(10)
            except BaseException as e:
                logger.error('Error in GenUnfo')
                logger.error(e)
                try:
                    traceback.print_tb(e)
                except BaseException:
                    pass


""" user the new method for getting follows"""
class GenUnfo2(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.user_id = user_utils.get_user_id(u)  # TODO: check null
        self.bot = d0.bot
        self.queue_to_unfo = d0.queue_to_unfo

    @retry
    def run(self):
        total = 0
        while True:
            try:
                for _id, _u in user_utils.get_all_follows_gen(self.bot, self.user_id):
                    logger.info('#%03d gen unfollow: %s', total, _u)
                    total += 1
                    self.queue_to_unfo.put(_id)
                time.sleep(10)
            except BaseException as e:
                logger.error('Error in GenUnfo')
                logger.error(e)
                try:
                    traceback.print_tb(e)
                except BaseException:
                    pass
                logger.info('sleeping for 60')
                time.sleep(60)

class DoUnfo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.bot = d0.bot
        self.queue_to_unfo = d0.queue_to_unfo

    @retry
    def run(self):
        daily_rate = 1000
        while True:
            try:
                f = self.queue_to_unfo.get()
                self.bot.unfollow(f)
                time.sleep(24 * 3600 / daily_rate)
            except BaseException as e:
                logger.error('Error in DoUnfo')
                logger.error(e)


class StealBase(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.uid = user_utils.get_user_id(u)
        self.bot = d0.bot
        self.queue_to_fo = d0.queue_to_fo
        self.conditions = user_config_reader.load_conditions()

    @retry
    def run(self):
        try:
            for id, name, msg in self.generate():
                logger.info("Candidate %s: %s", name.ljust(16), msg)
                if data.is_followed(self.u, id):
                    logger.debug("Already followed.")
                else:
                    if not Filter(name, self.conditions).apply():
                        logger.debug("Has not passed filter")
                    else:
                        logger.info("Good to steal! %s" % (name))
                        data.set_id_to_name(id, name)
                        self.queue_to_fo.put(id)
        except Exception as e:
            logger.error(repr(e))
            raise e


class Fofo(StealBase):
    def __init__(self, u):
        super().__init__(u)
        logger.info('user %s to FOFO', u)

    def generate(self):
        LEVEL_ONE_CAP = 200
        LEVEL_TWO_CAP = 100
        overall_cnt = 0
        for loop_cnt in itertools.count():
            for i, (id, name) in enumerate(user_utils.get_all_followers_gen(self.bot, self.uid, max=LEVEL_ONE_CAP)):
                for j, (_id, _name) in enumerate(user_utils.get_all_followers_gen(self.bot, id, max=LEVEL_TWO_CAP)):
                    overall_cnt += 1
                    message = "%d-th foer of %d-th foer(%s). Overall %d. loop %d." % (j,
                                                                                      i, name, overall_cnt, loop_cnt)
                    yield _id, _name, message


class StealSuperBrand(StealBase):
    def __init__(self, u):
        super().__init__(u)

    def generate(self):
        BATCH_SIZE = 1000
        BIG_LIST = ['instagram', 'apple', 'liuwenlw', 'london']
        brand_id_list = [user_utils.get_user_id(b) for b in BIG_LIST]
        for brand, brand_id in itertools.cycle(zip(BIG_LIST, brand_id_list)):
            for i, (id, name) in enumerate(user_utils.get_all_followers_gen(self.bot, brand_id, BATCH_SIZE)):
                yield id, name, "%d-th follower of super brand(%s)" % (i, brand)


class StealSimilarTo(StealBase):
    def __init__(self, u, seed_name):
        super().__init__(u)
        self.seed_name = seed_name

    def generate_star(self):
        MAX_QUEUE_SIZE = 1000
        star_queue = Queue()
        star = self.seed_name
        star_queue.put(star)
        next_stars = user_utils.related_users(self.bot, star)[:10]
        random.shuffle(next_stars)
        visited = set()
        logger.info('next stars %s', next_stars)
        for ns in next_stars:
            star_queue.put(ns)
            visited.add(ns)
        while True:
            star = star_queue.get()
            next_stars = user_utils.related_users(self.bot, star)[:50]
            random.shuffle(next_stars)
            for ns in next_stars:
                if ns not in visited and star_queue.qsize() < MAX_QUEUE_SIZE:
                    star_queue.put(ns)
                    visited.add(ns)
            yield star

    def generate(self):
        BATCH_SIZE = 500
        for star in self.generate_star():
            star_id = user_utils.get_user_id(star)  # TODO: check null
            yield star_id, star, "This is a star!"

            for i, (id, name) in enumerate(user_utils.get_all_followers_gen(self.bot, star_id, BATCH_SIZE)):
                yield id, name, "%d-th follower of star(%s)" % (i, star)


class Similar(StealSimilarTo):
    def __init__(self, u, seed_name):
        super().__init__(u, seed_name)

    def generate(self):
        for star in self.generate_star():
            star_id = user_utils.get_user_id(star)  # TODO: check null
            yield star_id, star, "This is a star!"


class StealFoers(StealBase):
    def __init__(self, u, steal_name):
        super().__init__(u)
        self.steal_name = steal_name
        self.steal_id = user_utils.get_user_id(steal_name)  # TODO: check null
        print('to steal user %s, id %d' % (steal_name, int(self.steal_id)))

    def generate(self):
        for i, (id, name) in enumerate(user_utils.get_all_followers_gen(self.bot, self.steal_id)):
            yield id, name, "The %d-th follower of user(%s)" % (i, self.steal_name)
        logger.info("Enumerated all followers of %s. My task is done!" % (self.steal_name))


class DoFo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.bot = d0.bot
        self.queue_to_fo = d0.queue_to_fo
        self.like_per_fo = d0.like_per_fo
        self.comment_pool = d0.comment_pool

    @retry
    def run(self):
        daily_rate = user_config_reader.load_follow_per_day()
        logger.info("Daily rate is %d", daily_rate)
        # TODO: extract all cooldown logic into separate module
        DEFAULT_LIKE_COOLDOWN = 100
        DEFAULT_COMMENT_COOLDOWN = 100
        like_cooldown = DEFAULT_LIKE_COOLDOWN
        like_cooldown_remain = 0
        comment_cooldown = DEFAULT_COMMENT_COOLDOWN
        comment_cooldown_remain = 0
        while True:
            try:
                if user_config_reader.load_incremental_daily_rate():
                    daily_rate += 0.01
                    logger.info("Daily rate increased to %f", daily_rate)

                # TODO: this is not UniqueQueue any more so possibly there's double-following, not a big deal
                # but can use a fix
                f = self.queue_to_fo.get()
                r = self.bot.follow(f)
                statsd.increment('naoshima.ig.follow', 1, tags=["user:" + self.u])
                if r.status_code == 200:
                    data.set_followed(self.u, f)
                else:
                    logger.error('Fail to follow, stats code: %d', r.status_code)
                    # TODO: cool down?
                    continue

                username = data.get_id_to_name(f)
                post_ids = user_utils.get_post_ids(username)

                # to like
                if len(post_ids) > self.like_per_fo:
                    post_ids = random.sample(post_ids, self.like_per_fo)
                if like_cooldown_remain <= 0:
                    for post_id in post_ids:
                        logger.info('like user(%s) post %d', username, int(post_id))
                        r = self.bot.like(post_id)
                        if r.status_code == 200:
                            like_cooldown = DEFAULT_LIKE_COOLDOWN
                            like_cooldown_remain = 0
                        else:
                            logger.info('fail to like. status code %d', r.status_code)
                            like_cooldown *= 1.2
                            logger.info('like cool down gap increased to %d', like_cooldown)
                            like_cooldown_remain = like_cooldown
                            logger.info('start like cool down for %d', like_cooldown)
                            break
                else:
                    logger.info('remain like cooldown %d', like_cooldown_remain)
                    like_cooldown_remain -= 1

                # to comment
                if post_ids and self.comment_pool:
                    post_id = random.choice(post_ids)
                    comment = random.choice(self.comment_pool)
                    logger.info('comment %s on %s', comment, str(post_id))
                    self.bot.comment(post_id, comment)

            except BaseException as e:
                logger.error('Error in DoFo')
                logger.error(e)
            finally:
                # slow down
                time.sleep(24 * 3600 / daily_rate)
