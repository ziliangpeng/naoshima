import pylru
import time
import requests

CACHED_USER_JSON = pylru.lrucache(1024)


def get_user_json(u):
    if u in CACHED_USER_JSON:
        return CACHED_USER_JSON[u]
    else:
        # TODO: find proper way to do rate limit
        time.sleep(1)  # initial delay
        url = 'https://www.instagram.com/%s/?__a=1' % u
        retry_delay = 5
        while True:
            r = requests.get(url)
            if r.status_code == 200:
                j = r.json()
                CACHED_USER_JSON[u] = j
                return j
            else:
                print(r.status_code)
                print('get json failed. sleeping for %d s' % (retry_delay))
                time.sleep(retry_delay)
                retry_delay = int(retry_delay * 1.2)  # exponentially increase delay
    # raise BaseException("Fail to get user json")
    print("Unable to get user json. Returning {}")
    return {}


def get_post_ids(u):
    j = get_user_json(u)
    posts = _json_path(j, ["user", "media", "nodes"])
    return [int(x["id"]) for x in posts]


def get_recent_post_epoch(u):
    j = get_user_json(u)
    posts = _json_path(j, ["user", "media", "nodes"])
    return posts and int(posts[0]["date"]) or -1


def get_biography(u):
    j = get_user_json(u)
    return _json_path(j, ["user", "biography"])


def get_user_id(u):
    j = get_user_json(u)
    return int(_json_path(j, ["user", "id"]))


def get_follows(u):
    j = get_user_json(u)
    return int(_json_path(j, ["user", "follows", "count"]))


def get_followed_by(u):
    j = get_user_json(u)
    return int(_json_path(j, ["user", "followed_by", "count"]))


def get_follow_counts(u):
    return get_followed_by(u), get_follows(u)


def _json_path(j, paths):
    for k in paths:
        if k in j:
            j = j[k]
        else:
            return None
    return j
