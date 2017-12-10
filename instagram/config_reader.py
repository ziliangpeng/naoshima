#!/usr/bin/python
# -*- coding: utf-8 -*-

import json5
import sys
from utils import _json_path


m = __import__(__name__)
METHODS = {
    'login': (["login"], None),
    'password': (["password"], None),
    # 'user_id': (["id"], None),
    'whitelist': (["whitelist"], []),
    'conditions': (["conditions"], {}),
    'max_follow': (["max_follow"], 5000),
    'min_follow': (["min_follow"], 1000),
    'commands': (["commands"], ["fofo", "unfo"]),
    'like_per_fo': (["like_per_fo"], 0),
    'comment_pool': (["comments"], []),  # [u"これは素晴らしい写真です", u"私はこの写真が好き"]
    'redis_host': (["redis", "host"], None),
    'redis_port': (["redis", "port"], None)
}

for k, (paths, default) in METHODS.items():
    def make_f(_paths, _default):
        return lambda: _load(_paths, _default)
    setattr(m, 'load_' + k, make_f(paths, default))


def load_secrets():
    return _load(["login"]), _load(["password"])


# def load_user_id():
#     return _load(["id"])


# def load_whitelist():
#     return _load(["whitelist"], [])


# def load_conditions():
#     return _load(["conditions"], {})


# def load_commands():
#     return _load(["commands"], ["fofo", "unfo"])


# def load_like_per_fo():
#     return _load(["like_per_fo"], 1)


# def load_comment_pool():
#     return _load(["comments"], [u"これは素晴らしい写真です", u"私はこの写真が好き"])


# def load_redis_host():
#     return _load(["redis", "host"])


# def load_redis_port():
#     return _load(["redis", "port"])


def _load(paths, default=None):
    with open('secret.local', 'r') as f:
        j = json5.loads(f.read())
        return _json_path(j, paths) or default


# publish a config file to redis
def publish(filename):
    import time
    import os.path
    import json
    import data

    with open(filename, 'r') as f:
        j = json5.loads(f.read())
        u = j.get('login')
        # TODO: check integrity of config

        if u == None:
            raise Exception("login should present")
        else:
            j['published_time'] = int(time.time())
            j['modified_time'] = int(os.path.getmtime(filename))
            print("The new config json is:\n")
            print(json.dumps(j, indent=4))

            if data.set_config(u, j):
                print("Upload complete")


if __name__ == '__main__':
    filename = sys.argv[1]
    publish(filename)
