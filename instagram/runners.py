import random
import datetime
import time

import utils
import secret_reader
import data
from threading import Thread


USER_ID = secret_reader.load_user_id()
WHITELIST_USER = secret_reader.load_whitelist()


class GenUnfo(Thread):
    def __init__(self, bot, queue_to_unfo, id_name_dict):
        Thread.__init__(self)
        self.bot = bot
        self.queue_to_unfo = queue_to_unfo
        self.id_name_dict = id_name_dict

    def run(self):
        while True:
            try:
                follows = utils.get_follows(self.bot, USER_ID)
                followers = utils.get_followers(self.bot, USER_ID)
                self.id_name_dict.update(follows)
                self.id_name_dict.update(followers)
                if len(follows) < 500:
                    time.sleep(60 * 10)
                    continue

                n = 100
                filtered_user_ids = filter(
                    lambda x: self.id_name_dict[x] not in WHITELIST_USER,
                    follows)
                to_unfo = random.sample(filtered_user_ids, n)
                random.shuffle(to_unfo)
                for i, f in enumerate(to_unfo):
                    print '%s: #%03d gen unfollow: %s' % \
                        (str(datetime.datetime.now()), i, self.id_name_dict[f])
                    self.queue_to_unfo.put(f)
                time.sleep(10)
            except BaseException as e:
                print 'Error in GenUnfo'
                print e


class DoUnfo(Thread):
    def __init__(self, bot, queue_to_unfo):
        Thread.__init__(self)
        self.bot = bot
        self.queue_to_unfo = queue_to_unfo

    def run(self):
        daily_rate = 1000
        while True:
            try:
                f = self.queue_to_unfo.get()
                self.bot.unfollow(f)
                time.sleep(24 * 3600 / daily_rate)
            except BaseException as e:
                print 'Error in DoUnfo'
                print e


class GenFo(Thread):
    def __init__(self, bot, queue_to_fo, id_name_dict, poked):
        Thread.__init__(self)
        self.bot = bot
        self.queue_to_fo = queue_to_fo
        self.id_name_dict = id_name_dict
        self.poked = poked

    def run(self):
        while True:
            try:
                n = 100
                fo_ids = random.sample(utils.find_fofo(self.bot, n, self.id_name_dict, self.poked), n)
                for i, f in enumerate(fo_ids):
                    print '%s: #%03d gen follow: %s' % \
                        (str(datetime.datetime.now()), i, self.id_name_dict[f])
                    self.queue_to_fo.put(f)
                    self.poked.add(f)
                time.sleep(10)
            except BaseException as e:
                print 'Error in GenFo'
                print e


class StealFoers(Thread):
    def __init__(self, bot, uid, queue_to_fo):
        Thread.__init__(self)
        self.bot = bot
        self.uid = uid
        self.queue_to_fo = queue_to_fo

    def run(self):
        i = 0
        for id, name in utils.get_all_followers_gen(self.bot, self.uid):
            i += 1
            if data.is_followed(id):
                print '%s: Skip %d-th follower %s(%s). Already followed.' % \
                    (str(datetime.datetime.now()), i, str(id), str(name))
            else:
                print '%s: Steal %d-th follower %s(%s)' % \
                    (str(datetime.datetime.now()), i, str(id), str(name))
                self.queue_to_fo.put(id)


class DoFo(Thread):
    def __init__(self, bot, queue_to_fo):
        Thread.__init__(self)
        self.bot = bot
        self.queue_to_fo = queue_to_fo

    def run(self):
        daily_rate = 999
        while True:
            try:
                f = self.queue_to_fo.get()
                self.bot.follow(f)
                data.follow(f)
                time.sleep(24 * 3600 / daily_rate)
            except BaseException as e:
                print 'Error in DoFo'
                print e
