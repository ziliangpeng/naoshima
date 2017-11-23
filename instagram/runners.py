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

WHITELIST_USER = secret_reader.load_whitelist()

# TODO:
# 1. follow followers' followers
# 2. keep following top brand's latest followers
# 3. keep following hashtag's most recent users


class GenUnfo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.username = u
        self.user_id = utils.get_user_id(u)
        self.bot = datas[u].bot
        self.queue_to_unfo = datas[u].queue_to_unfo

    def run(self):
        while True:
            try:
                follows = utils.get_follows(self.bot, self.user_id)
                # followers = utils.get_followers(self.bot, self.user_id)
                # self.id_name_dict.update(follows)
                # self.id_name_dict.update(followers)
                if len(follows) < 5000:
                    print('Only %d follows. Pause.' % len(follows))
                    time.sleep(60 * 30)
                    continue

                n = 100
                filtered_user_ids = [x for x in follows if data.get_id_to_name(x) not in WHITELIST_USER]
                to_unfo = random.sample(filtered_user_ids, n)
                random.shuffle(to_unfo)
                for i, f in enumerate(to_unfo):
                    print('%s: #%03d gen unfollow: %s' % \
                          (str(datetime.datetime.now()),
                           i, data.get_id_to_name(f)))
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


class Fofo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.bot = datas[u].bot
        self.queue_to_fo = datas[u].queue_to_fo
        print('user %s to FOFO' % u)

    def run(self):
        conditions = secret_reader.load_conditions()
        k = 0
        loop = 0
        while True:
            loop += 1
            try:
                i = 0
                for id, name in utils.get_all_followers_gen(self.bot, max=200):
                    i += 1
                    print('starting to steal from %d-th: %s' % (i, name))
                    j = 0
                    for _id, _name in utils.get_all_followers_gen(self.bot, uid=id, max=100):
                        j += 1
                        k += 1
                        print('inspecting %d-th foer(%s) of %d-th foer(%s), overall %d-th, %d-th loop' % \
                              (j, _name, i, name, k, loop))
                        if data.is_followed(self.u, _id):
                            print('Already followed.')
                        else:
                            if not Filter(_name, conditions).apply():
                                print('%s(%d) has not passed filter' % (_name, k))
                            else:
                                print('Steal follower %s' % (str(_name)))
                                data.set_id_to_name(_id, _name)
                                self.queue_to_fo.put(_id)
            except BaseException as e:
                print(e)
                pass


class StealSuperBrand(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.u = u
        self.bot = datas[u].bot
        self.queue_to_fo = datas[u].queue_to_fo

    def run(self):
        conditions = secret_reader.load_conditions()
        BIG_LIST = ['instagram', 'apple', 'liuwenlw', 'london']
        BATCH_SIZE = 1000
        while True:
            for brand in BIG_LIST:
                i = 0
                brand_id = utils.get_user_id(brand)
                for id, name in utils.get_all_followers_gen(self.bot, brand_id, BATCH_SIZE):
                    i += 1
                    print('inspecting %d-th foer of %s' % (i, brand))
                    if data.is_followed(self.u, id):
                        print('%s: Skip %d-th follower %s(%s). Already followed.' % \
                              (str(datetime.datetime.now()), i, str(id), str(name)))
                    else:
                        if not Filter(name, conditions).apply():
                            print('%s(%d) has not passed filter' % (name, i))
                        else:
                            print('%s: Steal %d-th follower %s(%s)' % \
                                  (str(datetime.datetime.now()), i, str(id), str(name)))
                            data.set_id_to_name(id, name)
                            self.queue_to_fo.put(id)


class StealFoers(Thread):
    def __init__(self, u, steal_name):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.steal_id = utils.get_user_id(steal_name)
        self.queue_to_fo = datas[u].queue_to_fo
        print('to steal user %s, id %d' % (steal_name, int(self.steal_id)))

    def run(self):
        conditions = secret_reader.load_conditions()
        i = 0
        skip_head = 0  # hack: skip something already processed
        for id, name in utils.get_all_followers_gen(self.bot, self.steal_id):
            i += 1
            if i < skip_head:
                print('skip head %d-th followers' % i)
                continue

            if data.is_followed(self.username, id):
                print('%s: Skip %d-th follower %s(%s). Already followed.' % \
                      (str(datetime.datetime.now()), i, str(id), str(name)))
            else:
                if not Filter(name, conditions).apply():
                    print('%s(%d) has not passed filter' % (name, i))
                else:
                    print('%s: Steal %d-th follower %s(%s)' % \
                          (str(datetime.datetime.now()), i, str(id), str(name)))
                    data.set_id_to_name(id, name)
                    self.queue_to_fo.put(id)


class DoFo(Thread):
    def __init__(self, u):
        Thread.__init__(self)
        self.username = u
        self.bot = datas[u].bot
        self.queue_to_fo = datas[u].queue_to_fo
        self.like_per_fo = datas[u].like_per_fo
        self.comment_pool = datas[u].comment_pool

    def run(self):
        daily_rate = 999
        # TODO: extract all cooldown logic into separate module
        DEFAULT_LIKE_COOLDOWN = 100
        DEFAULT_COMMENT_COOLDOWN = 100
        like_cooldown = DEFAULT_LIKE_COOLDOWN
        like_cooldown_remain = 0
        comment_cooldown = DEFAULT_COMMENT_COOLDOWN
        comment_cooldown_remain = 0
        while True:
            try:
                f = self.queue_to_fo.get()
                r = self.bot.follow(f)
                if r.status_code == 200:
                    data.set_followed(self.username, f)
                else:
                    print('fail to follow, stats code:', r.status_code)
                    # TODO: cool down?
                    continue

                username = data.get_id_to_name(f)
                post_ids = utils.get_post_ids(username)

                # to like
                if len(post_ids) > self.like_per_fo:
                    post_ids = random.sample(post_ids, self.like_per_fo)
                if like_cooldown_remain <= 0:
                    for post_id in post_ids:
                        print('like user(%s) post %d' % (username, int(post_id)))
                        r = self.bot.like(post_id)
                        if r.status_code == 200:
                            like_cooldown = DEFAULT_LIKE_COOLDOWN
                            like_cooldown_remain = 0
                        else:
                            print('fail to like. status code %d' % r.status_code)
                            # print(r.text)
                            like_cooldown *= 1.2
                            print('like cool down gap increased to', like_cooldown)
                            like_cooldown_remain = like_cooldown
                            print('start like cool down for', like_cooldown)
                            break
                else:
                    print('remain like cooldown', like_cooldown_remain)
                    like_cooldown_remain -= 1

                # to comment
                if post_ids and self.comment_pool:
                    post_id = random.choice(post_ids)
                    comment = random.choice(self.comment_pool)
                    print('comment %s on %s' % (comment, str(post_id)))
                    self.bot.comment(post_id, comment)

            except BaseException as e:
                print('Error in DoFo')
                print(e)
            finally:
                # slow down
                time.sleep(24 * 3600 / daily_rate)
