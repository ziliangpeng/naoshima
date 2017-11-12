import datetime
import random
import time
from threading import Thread

import data
import secret_reader
import utils
from filter import Filter

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
                # followers = utils.get_followers(self.bot, USER_ID)
                self.id_name_dict.update(follows)
                # self.id_name_dict.update(followers)
                if len(follows) < 5000:
                    print 'Only %d follows. Pause.' % len(follows)
                    time.sleep(60 * 30)
                    continue

                n = 100
                filtered_user_ids = filter(
                    lambda x: self.id_name_dict[x] not in WHITELIST_USER,
                    follows)
                to_unfo = random.sample(filtered_user_ids, n)
                random.shuffle(to_unfo)
                for i, f in enumerate(to_unfo):
                    print '%s: #%03d gen unfollow: %s' % \
                          (str(datetime.datetime.now()),
                           i, self.id_name_dict[f])
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
                fo_ids = random.sample(utils.find_fofo(
                    self.bot, n, self.id_name_dict, self.poked), n)
                for i, f in enumerate(fo_ids):
                    print '%s: #%03d gen follow: %s' % \
                          (str(datetime.datetime.now()),
                           i, self.id_name_dict[f])
                    self.queue_to_fo.put(f)
                    self.poked.add(f)
                time.sleep(10)
            except BaseException as e:
                print 'Error in GenFo'
                print e


class StealFoers(Thread):
    def __init__(self, bot, uid, queue_to_fo, id_name_dict):
        Thread.__init__(self)
        self.bot = bot
        self.uid = uid
        self.queue_to_fo = queue_to_fo
        self.id_name_dict = id_name_dict

    def run(self):
        conditions = secret_reader.load_conditions()
        i = 0
        skip_head = 0  # hack: skip something already processed
        for id, name in utils.get_all_followers_gen(self.bot, self.uid):
            i += 1
            if i < skip_head:
                print 'skip head %d-th followers' % i
                continue

            if data.is_followed(id):
                print '%s: Skip %d-th follower %s(%s). Already followed.' % \
                      (str(datetime.datetime.now()), i, str(id), str(name))
            else:
                if not Filter(name, conditions).apply():
                    print '%s has not passed filter' % name
                else:
                    print '%s: Steal %d-th follower %s(%s)' % \
                          (str(datetime.datetime.now()), i, str(id), str(name))
                    self.id_name_dict[int(id)] = name
                    self.queue_to_fo.put(id)


class DoFo(Thread):
    def __init__(self, bot, queue_to_fo, id_name_dict):
        Thread.__init__(self)
        self.bot = bot
        self.queue_to_fo = queue_to_fo
        self.id_name_dict = id_name_dict

    def run(self):
        daily_rate = 999
        like_cooldown = 0
        while True:
            try:
                f = self.queue_to_fo.get()
                self.bot.follow(f)
                data.follow(f)
                username = self.id_name_dict[int(f)]
                post_ids = utils.get_post_ids(username)
                if like_cooldown == 0:
                    for post_id in post_ids[:3]:
                        print 'like user(%s) post %d' % (username, int(post_id))
                        r = self.bot.like(post_id)
                        if r.status_code != 200:
                            print 'fail to like. status code %d' % r.status_code
                            print r.text
                            print 'start like cooldown'
                            like_cooldown = 999
                            break
                else:
                    print 'remain like cooldown', like_cooldown
                    like_cooldown -= 1
                time.sleep(24 * 3600 / daily_rate)
            except BaseException as e:
                print 'Error in DoFo'
                print e
