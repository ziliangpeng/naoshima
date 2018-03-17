import json
import time
import requests
import data
import urllib.request
import urllib.parse
import urllib.error
from utils import _json_path


def get_user_json(u):
    cached_json = data.get_json_by_username(u)
    if cached_json is not None:
        # print('user %s json cached' % (u))
        return cached_json
    else:
        # TODO: find proper way to do rate limit
        time.sleep(1)  # initial delay
        url = 'https://www.instagram.com/%s/?__a=1' % u
        retry_delay = 5
        while True:
            r = requests.get(url)
            if r.status_code == 200:
                j = r.json()
                data.set_json_by_username(u, j)
                return j
            elif r.status_code == 429:
                print('status code', r.status_code)
                print('get json failed. sleeping for %d s' % (retry_delay))
                time.sleep(retry_delay)
                retry_delay = int(retry_delay * 1.2)  # exponentially increase delay
            else:
                print('status code', r.status_code)
                break
    # raise BaseException("Fail to get user json")
    print("Unable to get user json. Returning {}")
    return {}


QUERY_IDs = {
    'follows': 17874545323001329,
    'followers': 17851374694183129,
    'related_user': 17845312237175864,
    'saved_media': 17885113105037631,
}
DEFAULT_PAGINATION = 8000
QUERY = '{"id":"%s","first":%d}'
QUERY_WITH_CURSOR = '{"id":"%s","first":%d,"after":"%s"}'
INSTAGRAM_GRAPPHQL_QUERY = 'https://www.instagram.com/graphql/query/?query_id=%d&variables=%s'
# for URL decode: https://meyerweb.com/eric/tools/dencoder/

# related user url:
# https://www.instagram.com/graphql/query/?query_id=17845312237175864&variables=%7B%22id%22%3A%225261744%22%7D

# saved media url:
# https://www.instagram.com/graphql/query/?query_id=17885113105037631&variables=%7B%22id%22%3A%226575470602%22%2C%22first%22%3A12%2C%22after%22%3A%22AQDiSrivWlEL4I0UzW00KCMhigZwMiPWpYimS2kihXKea8vIRfRaQR40RyTYA0BLHUxaolww2Li3-omro1cwi7kgoM8G5IK3925HrphwyawHFg%22%7D


def make_query_cursor(uid, paginate=DEFAULT_PAGINATION, cursor=""):
    return QUERY_WITH_CURSOR % (str(uid), int(paginate), str(cursor))


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

    url = INSTAGRAM_GRAPPHQL_QUERY % \
        (QUERY_IDs['saved_media'], urllib.parse.quote_plus(make_query_cursor(uid, paginate=200)))
    r = bot.s.get(url)
    if r.status_code != 200:
        return []
    j = r.json()
    nodes = [e["node"] for e in j["data"]["user"]["edge_saved_media"]["edges"]]
    return [make_media(n) for n in nodes]


""" special method, not simple reading. """


def get_follows(bot, uid):
    # TODO: this method need refactoring
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
            i = f["node"]["id"]
            u = f["node"]["username"]
            data.set_id_to_name(i, u)
            ret[i] = u
        return ret
    raise BaseException("Fail to get follows")


""" this is a special method """


def get_all_followers_gen(bot, uid, max=0):
    # TODO: require refactoring
    count = 0
    cursor = ""
    while True:
        while True:
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
                if max != 0 and count >= max:
                    return
                yield f["node"]["id"], f["node"]["username"]
                count += 1


def related_users(bot, u):
    # example url:
    # https://www.instagram.com/graphql/query/?query_id=17845312237175864&variables=%7B%22id%22%3A%225261744%22%7D
    uid = get_user_id(u)
    variables = make_query_cursor(uid)
    url = INSTAGRAM_GRAPPHQL_QUERY % \
        (QUERY_IDs['related_user'],
         urllib.parse.quote_plus(make_query_cursor(uid)))
    r = bot.s.get(url)
    if r.status_code == 200:
        j = r.json()
        return [n["node"]["username"] for n in j["data"]["user"]["edge_chaining"]["edges"]]
    else:
        return []


def get_post_ids(u):
    j = get_user_json(u)
    posts = _json_path(j, ['graphql', "user", "media", "nodes"])
    return [int(x["id"]) for x in posts]


""" Returns recent epoch date or -1 """


def get_recent_post_epoch(u):
    j = get_user_json(u)
    posts = _json_path(j, ['graphql', "user", "media", "nodes"])
    return posts and int(posts[0]["date"]) or -1


def get_biography(u):
    j = get_user_json(u)
    return _json_path(j, ['graphql', "user", "biography"])


def get_user_id(u):
    j = get_user_json(u)
    return int(_json_path(j, ['graphql', "user", "id"]))


def get_follows_count(u):
    j = get_user_json(u)
    return int(_json_path(j, ['graphql', "user", "follows", "count"]))


def get_followed_by_count(u):
    j = get_user_json(u)
    return int(_json_path(j, ['graphql', "user", "followed_by", "count"]))


def get_follow_counts(u):
    return get_followed_by_count(u), get_follows_count(u)


if __name__ == '__main__':
    # tests
    import auth
    b = auth.auth()
    u = 'instagram'
    print(u)
    while True:
        u = related_users(b, u)[0]
        print('-->', u)
        time.sleep(10)
