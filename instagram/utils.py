from typing import List


def _check_not_null(o):
    if o is None:
        raise BaseException("object cannot be None")
    return o


def _json_path(j: str, paths: List[str]):
    _check_not_null(j)
    for k in paths:
        if k in j:
            j = j[k]
        else:
            return None
    return j


""" Below is legacy utils. need to be removed."""
# import json
# import random
# import time
# import urllib.request, urllib.parse, urllib.error
# from user_utils import get_user_json
# import data
#
# import pylru
# import requests
#
# import secret_reader
#
# # for URL decode: https://meyerweb.com/eric/tools/dencoder/
#
#
# QUERY_IDs = {
#     'follows': 17874545323001329,
#     'followers': 17851374694183129,
# }
# DEFAULT_PAGINATION = 8000
# QUERY = '{"id":"%s","first":%d}'
# QUERY_WITH_CURSOR = '{"id":"%s","first":%d,"after":"%s"}'
# INSTAGRAM_GRAPPHQL_QUERY = 'https://www.instagram.com/graphql/query/?query_id=%d&variables=%s'
#
# USER_ID = secret_reader.load_user_id()
#
#
# def make_query_cursor(uid=USER_ID, paginate=DEFAULT_PAGINATION, cursor=""):
#     return QUERY_WITH_CURSOR % (str(uid), int(paginate), str(cursor))
#
#
# def map_user_id(user):
#     return user[0]
#
#
# def map_user_name(user):
#     return user[1]
#
#
# def get_follows(bot, uid=USER_ID):
#     time.sleep(3)  # initial delay
#     for retry in range(5):
#         time.sleep(2)  # retry delay
#         url = INSTAGRAM_GRAPPHQL_QUERY % \
#               (QUERY_IDs['follows'], urllib.parse.quote_plus(make_query_cursor(uid)))
#         r = bot.s.get(url)
#         if r.status_code != 200:
#             print('error in get follows, error code', r.status_code)
#             time.sleep(2)
#             continue
#         all_data = json.loads(r.text)
#         follows = all_data["data"]["user"]["edge_follow"]["edges"]
#         ret = {}
#         for f in follows:
#             i = f["node"]["id"]
#             u = f["node"]["username"]
#             data.set_id_to_name(i, u)
#             ret[i] = u
#         return ret
#     raise BaseException("Fail to get follows")
#
#
# def get_all_followers_gen(bot, uid=USER_ID, max=0):
#     count = 0
#     cursor = ""
#     while True:
#         while True:
#             time.sleep(3)  # initial delay
#             url = INSTAGRAM_GRAPPHQL_QUERY % \
#                   (QUERY_IDs['followers'],
#                    urllib.parse.quote_plus(make_query_cursor(uid, 500, cursor)))
#             r = bot.s.get(url)
#             if r.status_code != 200:
#                 print('error in get followers, error code', r.status_code)
#                 time.sleep(10)  # retry delay
#                 continue
#             all_data = json.loads(r.text)
#             followers = all_data["data"]["user"]["edge_followed_by"]["edges"]
#             if len(followers) == 0:
#                 return
#             cursor = all_data["data"]["user"]["edge_followed_by"]["page_info"]["end_cursor"]
#             for f in followers:
#                 if max != 0 and count >= max:
#                     return
#                 yield f["node"]["id"], f["node"]["username"]
#                 count += 1
#
#
# def get_all_followers(bot):
#     ret = {}
#     for k, v in get_all_followers_gen(bot):
#         ret[k] = v
#     return ret
#
#
# def get_post_ids(u):
#     try:
#         j = get_user_json(u)
#         posts = j["user"]["media"]["nodes"]
#         result = []
#         for post in posts:
#             result.append(post["id"])
#         return result
#     except BaseException:
#         return []
#
#
# def get_recent_post_epoch(u, default=None):
#     try:
#         j = get_user_json(u)
#         posts = j["user"]["media"]["nodes"]
#         if len(posts) == 0:
#             return -1
#         else:
#             return int(posts[0]["date"])
#     except BaseException as e:
#         print(e)
#         if default is None:
#             raise e
#         else:
#             return default
#
#
# def get_biography(u, default=''):
#     try:
#         j = get_user_json(u)
#         bio = j["user"]["biography"]
#         return bio
#     except BaseException as e:
#         print(e)
#         if default is None:
#             raise e
#         else:
#             return default
#
#
# def get_user_id(u, default=''):
#     try:
#         j = get_user_json(u)
#         id = j["user"]["id"]
#         return id
#     except BaseException as e:
#         print(e)
#         if default is None:
#             raise e
#         else:
#             return default
#
#
# def get_follow_counts(u, default=None):
#     try:
#         j = get_user_json(u)
#         return int(j["user"]["followed_by"]["count"]), int(j["user"]["follows"]["count"])
#     except BaseException as e:
#         if default is None:
#             raise e
#         else:
#             return default
#
#
# if __name__ == '__main__':
#     import auth
#
#     bot = auth.auth()
#     i = 0
#     for id, name in get_all_followers_gen(bot):
#         i += 1
#         print(i, id, name)
