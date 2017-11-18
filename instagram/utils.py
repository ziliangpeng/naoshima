import json
import random
import time
import urllib.request, urllib.parse, urllib.error
from user_utils import get_user_json

import pylru
import requests

import secret_reader

# for URL decode: https://meyerweb.com/eric/tools/dencoder/


QUERY_IDs = {
    'follows': 17874545323001329,
    'followers': 17851374694183129,
}
DEFAULT_PAGINATION = 8000
QUERY = '{"id":"%s","first":%d}'
QUERY_WITH_CURSOR = '{"id":"%s","first":%d,"after":"%s"}'
INSTAGRAM_GRAPPHQL_QUERY = 'https://www.instagram.com/graphql/query/?query_id=%d&variables=%s'

USER_ID = secret_reader.load_user_id()
CACHED_USER_JSON = pylru.lrucache(1024)


def make_query_cursor(uid=USER_ID, paginate=DEFAULT_PAGINATION, cursor=""):
    return QUERY_WITH_CURSOR % (str(uid), int(paginate), str(cursor))


def map_user_id(user):
    return user[0]


def map_user_name(user):
    return user[1]


def get_follows(bot, uid=USER_ID):
    time.sleep(3)  # initial delay
    for retry in range(5):
        time.sleep(2)  # retry delay
        url = INSTAGRAM_GRAPPHQL_QUERY % \
              (QUERY_IDs['follows'], urllib.parse.quote_plus(make_query_cursor(uid)))
        r = bot.s.get(url)
        if r.status_code != 200:
            print('error in get follows, error code', r.status_code)
            time.sleep(2)
            continue
        all_data = json.loads(r.text)
        follows = all_data["data"]["user"]["edge_follow"]["edges"]
        ret = {}
        for f in follows:
            ret[f["node"]["id"]] = f["node"]["username"]
        return ret
    raise BaseException("Fail to get follows")


# def get_followers(bot, uid=USER_ID):
#     time.sleep(3)  # initial delay
#     for retry in xrange(5):
#         time.sleep(2)  # retry delay
#         url = INSTAGRAM_GRAPPHQL_QUERY % \
#               (QUERY_IDs['followers'], urllib.quote_plus(make_query_cursor(uid)))
#         r = bot.s.get(url)
#         if r.status_code != 200:
#             print 'error in get followers, error code', r.status_code
#             continue
#         all_data = json.loads(r.text)
#         followers = all_data["data"]["user"]["edge_followed_by"]["edges"]
#         ret = {}
#         for f in followers:
#             ret[f["node"]["id"]] = f["node"]["username"]
#         return ret
#     raise BaseException("Fail to get followers")


def get_all_followers_gen(bot, uid=USER_ID, max=0):
    count = 0
    cursor = ""
    while True:
        while True:
            if max != 0 and count >= max:
                return
            time.sleep(3)  # initial delay
            url = INSTAGRAM_GRAPPHQL_QUERY % \
                  (QUERY_IDs['followers'],
                   urllib.parse.quote_plus(make_query_cursor(uid, 500, cursor)))
            r = bot.s.get(url)
            if r.status_code != 200:
                print('error in get followers, error code', r.status_code)
                time.sleep(10)  # retry delay
                continue
            all_data = json.loads(r.text)
            followers = all_data["data"]["user"]["edge_followed_by"]["edges"]
            if len(followers) == 0:
                return
            cursor = all_data["data"]["user"]["edge_followed_by"]["page_info"]["end_cursor"]
            for f in followers:
                yield f["node"]["id"], f["node"]["username"]
                count += 1


def get_all_followers(bot):
    ret = {}
    for k, v in get_all_followers_gen(bot):
        ret[k] = v
    return ret


# def find_fofo(bot, n, id_name_dict, poked):
#     follows = get_follows(bot)
#     followers = get_followers(bot)
#     id_name_dict.update(follows)
#     id_name_dict.update(followers)
#     fo_list = set()
#     while len(fo_list) < n:
#         chosen_foer = random.choice(list(follows.keys()))
#         foer_foer = get_followers(bot, chosen_foer)
#         id_name_dict.update(foer_foer)
#         fo_list |= set(foer_foer) - set(follows) - poked
#         fo_list.discard(USER_ID)
#     return fo_list


# def get_user_json(username):
#     if username in CACHED_USER_JSON:
#         return CACHED_USER_JSON[username]
#     else:
#         time.sleep(3)  # initial delay
#         url = 'https://www.instagram.com/%s/?__a=1' % username
#         for retry in range(7):
#             r = requests.get(url)
#             if r.status_code == 200:
#                 j = r.json()
#                 CACHED_USER_JSON[username] = j
#                 return j
#             else:
#                 time.sleep(5)
#                 continue
#     raise BaseException("Fail to get user json")


def get_post_ids(username):
    try:
        j = get_user_json(username)
        posts = j["user"]["media"]["nodes"]
        result = []
        for post in posts:
            result.append(post["id"])
        return result
    except BaseException:
        return []


def get_recent_post_epoch(username, default=None):
    try:
        j = get_user_json(username)
        posts = j["user"]["media"]["nodes"]
        if len(posts) == 0:
            return -1
        else:
            return int(posts[0]["date"])
    except BaseException as e:
        print(e)
        if default is None:
            raise e
        else:
            return default


def get_biography(username, default=''):
    try:
        j = get_user_json(username)
        bio = j["user"]["biography"]
        return bio
    except BaseException as e:
        print(e)
        if default is None:
            raise e
        else:
            return default


def get_user_id(username, default=''):
    try:
        j = get_user_json(username)
        id = j["user"]["id"]
        return id
    except BaseException as e:
        print(e)
        if default is None:
            raise e
        else:
            return default


def get_follow_counts(username, default=None):
    try:
        j = get_user_json(username)
        return int(j["user"]["followed_by"]["count"]), int(j["user"]["follows"]["count"])
    except BaseException as e:
        if default is None:
            raise e
        else:
            return default


if __name__ == '__main__':
    import auth

    bot = auth.auth()
    i = 0
    for id, name in get_all_followers_gen(bot):
        i += 1
        print(i, id, name)
