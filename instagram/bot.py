import sys
import auth
import secret_reader
import random
import urllib
import time
import json
import datetime
from threading import Thread
from model import UniqueQueue


WHITELIST_USER = secret_reader.load_whitelist()
FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50
USER_ID = secret_reader.load_user_id()

bot = auth.auth()
queue_to_fo = UniqueQueue(FO_QUEUE_SIZE)
queue_to_unfo = UniqueQueue(UNFO_QUEUE_SIZE)
poked = set()  # ppl I've followed before


###########  Runnables  ################

def gen_unfo():
    while True:
        try:
            follows = get_follows(USER_ID)
            followers = get_followers(USER_ID)
            id_name_dict.update(follows)
            id_name_dict.update(followers)

            user_id_not_whitelisted = lambda x: id_name_dict[x] not in WHITELIST_USER
            n = 100
            to_unfo = random.sample(filter(user_id_not_whitelisted, follows), n)
            random.shuffle(to_unfo)
            for i, f in enumerate(to_unfo):
                print '%s: #%03d gen unfollow: %s' % (str(datetime.datetime.now()), i, id_name_dict[f])
                queue_to_unfo.put(f)
            time.sleep(10)
        except BaseException as e:
            print e

def do_unfo():
    daily_rate = 1000
    while True:
        try:
            f = queue_to_unfo.get()
            bot.unfollow(f)
            time.sleep(24 * 3600 / daily_rate)
        except BaseException as e:
            print e

def gen_fo():
    while True:
        try:
            n = 100
            fo_ids = random.sample(find_fofo(n), n)
            for i, f in enumerate(fo_ids):
                print '%s: #%03d gen follow: %s' % (str(datetime.datetime.now()), i, id_name_dict[f])
                queue_to_fo.put(f)
                poked.add(f)
            time.sleep(10)
        except BaseException as e:
            print e

def do_fo():
    daily_rate = 999
    while True:
        try:
            f = queue_to_fo.get()
            bot.follow(f)
            time.sleep(24 * 3600 / daily_rate)
        except BaseException as e:
            print e

if __name__ == '__main__':
    for cmd in sys.argv[1:]:
        if cmd == 'fo':
            Thread(target = gen_fo).start()
            Thread(target = do_fo).start()
        elif cmd == 'unfo':
            Thread(target = gen_unfo).start()
            Thread(target = do_unfo).start()

    while True:
        time.sleep(1)
