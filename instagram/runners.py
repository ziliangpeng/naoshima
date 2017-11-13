import datetime
import random
import time
from threading import Thread

import data
import secret_reader
import utils
from filter import Filter
import data_repo
from data_repo import datas

USER_ID = secret_reader.load_user_id()
WHITELIST_USER = secret_reader.load_whitelist()


class GenUnfo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.queue_to_unfo = datas[u].queue_to_unfo
        self.id_name_dict = data_repo.id_name_dict

    def run(self):
        while True:
            try:
                follows = utils.get_follows(self.bot, USER_ID)
                # followers = utils.get_followers(self.bot, USER_ID)
                self.id_name_dict.update(follows)
                # self.id_name_dict.update(followers)
                if len(follows) < 5000:
                    print('Only %d follows. Pause.' % len(follows))
                    time.sleep(60 * 30)
                    continue

                n = 100
                filtered_user_ids = [x for x in follows if self.id_name_dict[x] not in WHITELIST_USER]
                to_unfo = random.sample(filtered_user_ids, n)
                random.shuffle(to_unfo)
                for i, f in enumerate(to_unfo):
                    print('%s: #%03d gen unfollow: %s' % \
                          (str(datetime.datetime.now()),
                           i, self.id_name_dict[f]))
                    self.queue_to_unfo.put(f)
                time.sleep(10)
            except BaseException as e:
                print('Error in GenUnfo')
                print(e)


class DoUnfo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.queue_to_unfo = datas[u].queue_to_unfo

    def run(self):
        daily_rate = 1000
        while True:
            try:
                f = self.queue_to_unfo.get()
                self.bot.unfollow(f)
                time.sleep(24 * 3600 / daily_rate)
            except BaseException as e:
                print('Error in DoUnfo')
                print(e)


class GenFo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.queue_to_fo = datas[u].queue_to_fo
        self.id_name_dict = data_repo.id_name_dict
        self.poked = datas[u].poked

    # def run(self):
    #     while True:
    #         try:
    #             n = 100
    #             fo_ids = random.sample(utils.find_fofo(
    #                 self.bot, n, self.id_name_dict, self.poked), n)
    #             for i, f in enumerate(fo_ids):
    #                 print '%s: #%03d gen follow: %s' % \
    #                       (str(datetime.datetime.now()),
    #                        i, self.id_name_dict[f])
    #                 self.queue_to_fo.put(f)
    #                 self.poked.add(f)
    #             time.sleep(10)
    #         except BaseException as e:
    #             print 'Error in GenFo'
    #             print e


class StealFoers(Thread):
    def __init__(self, u, steal_id):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.steal_id = steal_id
        self.queue_to_fo = datas[u].queue_to_fo
        self.id_name_dict = data_repo.id_name_dict

    def run(self):
        conditions = secret_reader.load_conditions()
        i = 0
        skip_head = 0  # hack: skip something already processed
        for id, name in utils.get_all_followers_gen(self.bot, self.steal_id):
            i += 1
            if i < skip_head:
                print('skip head %d-th followers' % i)
                continue

            if data.is_followed(id):
                print('%s: Skip %d-th follower %s(%s). Already followed.' % \
                      (str(datetime.datetime.now()), i, str(id), str(name)))
            else:
                if not Filter(name, conditions).apply():
                    print('%s has not passed filter' % name)
                else:
                    print('%s: Steal %d-th follower %s(%s)' % \
                          (str(datetime.datetime.now()), i, str(id), str(name)))
                    self.id_name_dict[int(id)] = name
                    self.queue_to_fo.put(id)


class DoFo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.queue_to_fo = datas[u].queue_to_fo
        self.id_name_dict = data_repo.id_name_dict
        self.like_per_fo = datas[u].like_per_fo
        self.comment_pool = datas[u].comment_pool

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

                # to like
                if len(post_ids) > self.like_per_fo:
                    post_ids = random.sample(post_ids, self.like_per_fo)
                if like_cooldown == 0:
                    for post_id in post_ids:
                        print('like user(%s) post %d' % (username, int(post_id)))
                        r = self.bot.like(post_id)
                        if r.status_code != 200:
                            print('fail to like. status code %d' % r.status_code)
                            print(r.text)
                            print('start like cooldown')
                            like_cooldown = 999
                            break
                else:
                    print('remain like cooldown', like_cooldown)
                    like_cooldown -= 1

                # to comment
                if post_ids:
                    post_id = random.choice(post_ids)
                    comment = random.choice(self.comment_pool)
                    print('comment %s on %s' % (comment, str(post_id)))
                    self.bot.comment(post_id, comment)

                # slow down
                time.sleep(24 * 3600 / daily_rate)
            except BaseException as e:
                print('Error in DoFo')
                print(e)
