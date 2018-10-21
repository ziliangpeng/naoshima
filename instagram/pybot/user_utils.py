import json
import time
from functools import lru_cache

import requests
import fetcher
import data
# import data_repo
import query_hash
import urllib.request
import urllib.parse
import urllib.error

import user_config_reader
from dd import sd
from utils import _json_path, rate_limit_get
from logs import logger


from_user = user_config_reader.load_secrets()[0]

@sd.timed(metric='naoshima.ig.get_user_json_cached.time', tags=['user:' + from_user])
@lru_cache(maxsize=32)
def get_user_json(u):
    cached_json = data.get_json_by_username(u)
    if cached_json is not None:
        sd.increment('naoshima.ig.get_user.cache.hit', tags=['user:' + from_user])
        return cached_json

    status_code, j = fetcher.get_user_json(u)
    if status_code == 200:
        data.set_json_by_username(u, j)
        return j
    else:
        logger.warn("Unable to get user json. status code %d. Returning {}", status_code)
        return {}


QUERY_IDs = {
    'follows': 17874545323001329,
    'followers': 17851374694183129,
    'related_user': 17845312237175864,
    'saved_media': 17885113105037631,
}
QUERY_HASHs = {
    'follows': '58712303d941c6855d4e888c5f0cd22f',
    'followers': '37479f2b8209594dde7facb0d904896a',
    'related_user': 'fake',
    'saved_media': 'fake',
}
DEFAULT_PAGINATION = 50
QUERY = '{"id":"%s","first":%d}'
QUERY_WITH_CURSOR = '{"id":"%s","first":%d,"after":"%s"}'
PROFILE_QUERY = '{"user_id":"%s","include_chaining":true,"include_reel":true,"include_suggested_users":false,"include_logged_out_extras":false,"include_highlight_reels":false}'
INSTAGRAM_GRAPPHQL_ID_QUERY = 'https://www.instagram.com/graphql/query/?query_id=%d&variables=%s'
INSTAGRAM_GRAPPHQL_HASH_QUERY = 'https://www.instagram.com/graphql/query/?query_hash=%s&variables=%s'
# for URL decode: https://meyerweb.com/eric/tools/dencoder/

# follow url (query_hash)
# https://www.instagram.com/graphql/query/?query_hash=58712303d941c6855d4e888c5f0cd22f&variables=%7B%22id%22%3A%222288001113%22%2C%22first%22%3A24%7D

# related user url:
# https://www.instagram.com/graphql/query/?query_id=17845312237175864&variables=%7B%22id%22%3A%225261744%22%7D

# saved media url:
# https://www.instagram.com/graphql/query/?query_id=17885113105037631&variables=%7B%22id%22%3A%226575470602%22%2C%22first%22%3A12%2C%22after%22%3A%22AQDiSrivWlEL4I0UzW00KCMhigZwMiPWpYimS2kihXKea8vIRfRaQR40RyTYA0BLHUxaolww2Li3-omro1cwi7kgoM8G5IK3925HrphwyawHFg%22%7D


def make_query_cursor(uid, paginate=DEFAULT_PAGINATION, cursor=""):
    return QUERY_WITH_CURSOR % (str(uid), int(paginate), str(cursor))


def make_profile_query(uid):
    return PROFILE_QUERY % (uid)


def get_saved_medias(bot, uid):
    class Media:
        def __init__(self, photo_id, code, typename, url, caption):
            self.photo_id = photo_id
            self.code = code
            self.typename = typename
            self.url = url
            self.caption = caption

    def make_media(n):
        photo_id = n["id"]
        code = n["shortcode"]
        typename = n["__typename"]
        url = n["display_url"]
        try:
            caption = n["edge_media_to_caption"]["edges"][0]["node"]["text"]
        except BaseException:
            caption = ""
        return Media(photo_id, code, typename, url, caption)

    url = INSTAGRAM_GRAPPHQL_ID_QUERY % \
        (QUERY_IDs['saved_media'], urllib.parse.quote_plus(make_query_cursor(uid, paginate=50)))
    r = rate_limit_get(bot.s, url)
    if r.status_code != 200:
        return []
    j = r.json()
    nodes = [e["node"] for e in j["data"]["user"]["edge_saved_media"]["edges"]]
    return [make_media(n) for n in nodes]


""" special method, not simple reading. """


def get_all_followers_gen(bot, uid, max=0):
    count = 0
    cursor = ""
    while True:
        url = INSTAGRAM_GRAPPHQL_HASH_QUERY % \
            (query_hash.follower_hash(bot.s),
             urllib.parse.quote_plus(make_query_cursor(uid, 50, cursor)))
        logger.info("followers url %s", url)
        r = rate_limit_get(bot.s, url)
        if r.status_code == 400:
            # TODO: temporary hack, work around end of cursor
            return
        elif r.status_code != 200:
            logger.warn("error in get followers, error code %d", r.status_code)
            raise BaseException("Error in getting followers")

        all_data = json.loads(r.text)
        followers = all_data["data"]["user"]["edge_followed_by"]["edges"]
        if len(followers) == 0:
            return
        cursor = all_data["data"]["user"]["edge_followed_by"]["page_info"]["end_cursor"]
        logger.info("next cursor is %s" % (str(cursor)))
        for f in followers:
            if max != 0 and count >= max:
                return
            yield f["node"]["id"], f["node"]["username"]
            count += 1


def get_all_follows_gen(bot, uid, max=0):
    count = 0
    cursor = ""
    while True:
        url = INSTAGRAM_GRAPPHQL_HASH_QUERY % \
            (query_hash.following_hash(bot.s),
             urllib.parse.quote_plus(make_query_cursor(uid, 50, cursor)))
        logger.info("following url %s", url)
        r = rate_limit_get(bot.s, url)
        if r.status_code != 200:
            logger.warn("error in get following, error code %d", r.status_code)
            continue
            raise BaseException("Error in getting follows")
        all_data = json.loads(r.text)
        followers = all_data["data"]["user"]["edge_follow"]["edges"]
        if len(followers) == 0:
            return
        cursor = all_data["data"]["user"]["edge_follow"]["page_info"]["end_cursor"]
        for f in followers:
            if max != 0 and count >= max:
                return
            yield f["node"]["id"], f["node"]["username"]
            count += 1


def get_related_users_gen(bot, uid):
    # example url:
    # https://www.instagram.com/graphql/query/?query_id=17845312237175864&variables=%7B%22id%22%3A%225261744%22%7D
    url = INSTAGRAM_GRAPPHQL_HASH_QUERY % \
        (query_hash.profile_hash(bot.s),
         urllib.parse.quote_plus(make_profile_query(uid)))
    logger.info("related users gen url %s", url)
    r = rate_limit_get(bot.s, url)
    if r.status_code == 200:
        j = r.json()
        for n in j["data"]["user"]["edge_chaining"]["edges"]:
            yield n['node']['id'], n["node"]["username"]
    else:
        return


def related_users(bot, u):
    # example url:
    # https://www.instagram.com/graphql/query/?query_id=17845312237175864&variables=%7B%22id%22%3A%225261744%22%7D
    uid = get_user_id(u)
    variables = make_query_cursor(uid)
    url = INSTAGRAM_GRAPPHQL_HASH_QUERY % \
        (query_hash.profile_hash(bot.s),
         urllib.parse.quote_plus(make_profile_query(uid)))
    logger.info("related users url %s", url)
    r = rate_limit_get(bot.s, url)
    if r.status_code == 200:
        j = r.json()
        return [n["node"]["username"] for n in j["data"]["user"]["edge_chaining"]["edges"]]
    else:
        return []


def get_post_ids(u):
    j = get_user_json(u)
    posts = _json_path(j, ['graphql', "user", "edge_owner_to_timeline_media", "edges"])
    return [int(x["node"]["id"]) for x in posts]


""" Returns recent epoch date or -1 """


def get_recent_post_epoch(u):
    j = get_user_json(u)
    # TODO: user get_recent_posts to get posts
    posts = _json_path(j, ['graphql', "user", "edge_owner_to_timeline_media", "edges"])
    return posts and int(posts[0]["node"]["taken_at_timestamp"]) or -1


def get_recent_posts(u):
    j = get_user_json(u)
    posts = _json_path(j, ['graphql', "user", "edge_owner_to_timeline_media", "edges"])
    return posts


def get_biography(u):
    j = get_user_json(u)
    return _json_path(j, ['graphql', "user", "biography"])


def get_user_id(u):
    j = get_user_json(u)
    user_id = _json_path(j, ['graphql', "user", "id"])
    if user_id is None:
        logger.error("Error getting id for user %s", u)
        return None
    return int(user_id)


def get_follows_count(u):
    j = get_user_json(u)
    return int(_json_path(j, ['graphql', "user", "edge_follow", "count"]))


def get_followed_by_count(u):
    j = get_user_json(u)
    return int(_json_path(j, ['graphql', "user", "edge_followed_by", "count"]))


def get_follow_counts(u):
    return get_followed_by_count(u), get_follows_count(u)


# d0 = data_repo.d0
# d0.user_id = get_user_id(d0.u)  # TODO: check null

if __name__ == '__main__':
    # tests
    import auth
    b = auth.auth()
    print('user id')
    u = 'bokehcume'
    uid = get_user_id(u)
    print(uid)
    print('followers')
    for id, u in get_all_followers_gen(b, uid):
        print(id, u)
        break

    print('follows')
    for id, u in get_all_follows_gen(b, uid):
        print(id, u)
        break

    print('related')
    for _id, _u in get_related_users_gen(b, uid):
        print(_id, _u)
        break
    # u = 'instagram'
    # print(u)
    # while True:
    #     u = related_users(b, u)[0]
    #     print('-->', u)
    #     time.sleep(10)
