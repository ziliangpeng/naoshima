import auth
import secret_reader
import random
import urllib
import time
import sys
import json
import datetime
from threading import Thread
from model import UniqueQueue


QUERY_IDs = {
    'follows': 17874545323001329,
    'followers': 17851374694183129,
}
WHITELIST_USER = secret_reader.load_whitelist()
FO_QUEUE_SIZE = 50
UNFO_QUEUE_SIZE = 50
USER_ID = secret_reader.load_user_id()

bot = auth.auth()
id_name_dict = {}
queue_to_fo = UniqueQueue(FO_QUEUE_SIZE)
queue_to_unfo = UniqueQueue(UNFO_QUEUE_SIZE)

# for URL decode: https://meyerweb.com/eric/tools/dencoder/

DEFAULT_PAGINATION = 8000
QUERY = '{"id":"%s","first":%d}'
INSTAGRAM_GRAPPHQL_QUERY = 'https://www.instagram.com/graphql/query/?query_id=%d&variables=%s'

poked = set() # ppl I've followed before

def make_query(uid=USER_ID, paginate=DEFAULT_PAGINATION):
    return QUERY % (str(uid), int(paginate))

def map_user_id(user):
    return user[0]

def map_user_name(user):
    return user[1]

def get_follows(uid=USER_ID):
    time.sleep(3) # initial delay
    for retry in xrange(5):
        time.sleep(2) # retry delay
        url = INSTAGRAM_GRAPPHQL_QUERY % (QUERY_IDs['follows'], urllib.quote_plus(make_query(uid)))
        r = bot.s.get(url)
        if r.status_code != 200:
            print 'error in get follows, error code', r.status_code
            time.sleep(2)
            continue
        all_data = json.loads(r.text)
        follows = all_data["data"]["user"]["edge_follow"]["edges"]
        ret = {}
        for f in follows:
            ret[f["node"]["id"]] = f["node"]["username"]
        return ret
    raise BaseException("Fail to get follows")

def get_followers(uid=USER_ID):
    time.sleep(3) # initial delay
    for retry in xrange(5):
        time.sleep(2) # retry delay
        url = INSTAGRAM_GRAPPHQL_QUERY % (QUERY_IDs['followers'], urllib.quote_plus(make_query(uid)))
        r = bot.s.get(url)
        if r.status_code != 200:
            print 'error in get followers, error code', r.status_code
            continue
        all_data = json.loads(r.text)
        followers = all_data["data"]["user"]["edge_followed_by"]["edges"]
        ret = {}
        for f in followers:
            ret[f["node"]["id"]] = f["node"]["username"]
        return ret
    raise BaseException("Fail to get followers")


""" Returns a list of ids to fo. """
def find_fofo(n):
    follows = get_follows()
    followers = get_followers()
    id_name_dict.update(follows)
    id_name_dict.update(followers)
    fo_list = set()
    while len(fo_list) < n:
        chosen_foer = random.choice(list(follows.keys()))
        foer_foer = get_followers(chosen_foer)
        id_name_dict.update(foer_foer)
        fo_list |= set(foer_foer) - set(follows) - poked
        fo_list.discard(USER_ID)
    return fo_list


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
