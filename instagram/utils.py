import time
import urllib
import json
import random
import secret_reader
# for URL decode: https://meyerweb.com/eric/tools/dencoder/


QUERY_IDs = {
    'follows': 17874545323001329,
    'followers': 17851374694183129,
}
DEFAULT_PAGINATION = 8000
QUERY = '{"id":"%s","first":%d}'
INSTAGRAM_GRAPPHQL_QUERY = 'https://www.instagram.com/graphql/query/?query_id=%d&variables=%s'

USER_ID = secret_reader.load_user_id()

id_name_dict = {}


def make_query(uid=USER_ID, paginate=DEFAULT_PAGINATION):
    return QUERY % (str(uid), int(paginate))


def map_user_id(user):
    return user[0]


def map_user_name(user):
    return user[1]


def get_follows(bot, uid=USER_ID):
    time.sleep(3)  # initial delay
    for retry in xrange(5):
        time.sleep(2)  # retry delay
        url = INSTAGRAM_GRAPPHQL_QUERY % \
            (QUERY_IDs['follows'], urllib.quote_plus(make_query(uid)))
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


def get_followers(bot, uid=USER_ID):
    time.sleep(3)  # initial delay
    for retry in xrange(5):
        time.sleep(2)  # retry delay
        url = INSTAGRAM_GRAPPHQL_QUERY % \
            (QUERY_IDs['followers'], urllib.quote_plus(make_query(uid)))
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