import datetime

import itertools

import logs
import random
import time
from threading import Thread
from retrying import retry
import traceback
import sys

import data
import user_config_reader
import user_utils
from filter import Filter
from queue import Queue
from data_repo import d0
from dd import m

WHITELIST_USER = user_config_reader.load_whitelist()

logger = logs.logger

# TODO:
# 1. follow followers' followers
# 2. keep following top brand's latest followers
# 3. keep following hashtag's most recent users

class InfinityTask(Thread):
    def __init__(self):
        Thread.__init__(self)
        self.delay = 0.1  # seconds

    @retry
    @logs.log_exception
    def run(self):
        while True:
            self.task()
            time.sleep(self.delay)


class GenUnfo2(InfinityTask):
    def __init__(self, u):
        InfinityTask.__init__(self)
        self.u = u
        self.user_id = user_utils.get_user_id(u)  # TODO: check null
        self.bot = d0.bot
        self.queue_to_unfo = d0.queue_to_unfo

        self.delay = 10
        self.total = 0

    def task(self):
        MIN_FOLLOW_THRESHOLD = 999
        follows = user_utils.get_follows_count(self.u)
        if MIN_FOLLOW_THRESHOLD > follows:
            logger.info("Only %d follows. Pause.", follows)
            time.sleep(60 * 30)
            return

        for _id, _u in user_utils.get_all_follows_gen(self.bot, self.user_id):
            if _u in WHITELIST_USER:
                continue
            logger.info('#%03d gen unfollow: %s', self.total, _u)
            self.total += 1
            self.queue_to_unfo.put(_id)


class DoUnfo(InfinityTask):
    def __init__(self, u):
        InfinityTask.__init__(self)
        self.u = u
        self.bot = d0.bot
        self.queue_to_unfo = d0.queue_to_unfo
        self.daily_rate = 1000
        self.delay = 24 * 3600 / self.daily_rate

    def task(self):
        f = self.queue_to_unfo.get()
        self.bot.unfollow(f)


class StealBase(InfinityTask):
    def __init__(self, u):
        InfinityTask.__init__(self)
        self.u = u
        self.uid = user_utils.get_user_id(u)
        self.bot = d0.bot
        self.queue_to_fo = d0.queue_to_fo
        self.conditions = user_config_reader.load_conditions()

    def task(self):
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
        BATCH_SIZE = 200
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


class DoFo(InfinityTask):
    def __init__(self, u):
        InfinityTask.__init__(self)
        self.u = u
        self.bot = d0.bot
        self.queue_to_fo = d0.queue_to_fo
        self.like_per_fo = d0.like_per_fo
        self.comment_pool = d0.comment_pool

        self.daily_rate = user_config_reader.load_follow_per_day()
        logger.info("Daily rate is %d", self.daily_rate)
        self.delay = 24 * 3600 / self.daily_rate

    def task(self):
        # TODO: extract all cooldown logic into separate module
        DEFAULT_LIKE_COOLDOWN = 100
        DEFAULT_COMMENT_COOLDOWN = 100
        like_cooldown = DEFAULT_LIKE_COOLDOWN
        like_cooldown_remain = 0
        comment_cooldown = DEFAULT_COMMENT_COOLDOWN
        comment_cooldown_remain = 0
        if user_config_reader.load_incremental_daily_rate():
            self.daily_rate += 0.01
            logger.info("Daily rate increased to %f", self.daily_rate)

        # TODO: this is not UniqueQueue any more so possibly there's double-following, not a big deal
        # but can use a fix
        f = self.queue_to_fo.get()
        logger.info("Follow " + f)
        r = self.bot.follow(f)
        m.followed()
        if r.status_code == 200:
            data.set_followed(self.u, f)
        else:
            logger.error('Fail to follow, stats code: %d', r.status_code)
            # TODO: cool down?
            return

        username = data.get_id_to_name(f)
        post_ids = user_utils.get_post_ids(username)

        # to like
        if len(post_ids) > self.like_per_fo:
            post_ids = random.sample(post_ids, self.like_per_fo)
        if like_cooldown_remain <= 0:
            for post_id in post_ids:
                logger.info('like user(%s) post %s', username, str(post_id))
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
