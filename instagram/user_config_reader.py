#!/usr/bin/python
# -*- coding: utf-8 -*-

import json5
import sys

from utils import _json_path


user_key = sys.argv[1]

m = __import__(__name__)
METHODS = {
    'login': (["login"], None),
    'password': (["password"], None),
    'whitelist': (["whitelist"], []),
    'conditions': (["conditions"], {}),
    'max_follows': (["max_follows"], 5000),
    'min_follows': (["min_follows"], 1000),
    'max_follow': (["max_follows"], 5000),  # deprecated, and backward-compatible
    'min_follow': (["min_follows"], 1000),  # deprecated, and backward-compatible
    'follow_per_day': (["follow_per_day"], 1000),
    'incremental_daily_rate': (["incremental_daily_rate"], False),
    'commands': (["commands"], ["fofo", "unfo"]),
    'like_per_fo': (["like_per_fo"], 0),
    'comment_pool': (["comments"], []),  # [u"これは素晴らしい写真です", u"私はこの写真が好き"]
}

for k, (paths, default) in METHODS.items():
    def make_f(_paths, _default):
        return lambda: _load([user_key] + _paths, _default)

    def make_redis_f(_paths, _default):
        return lambda: _load_from_redis(user_key, _paths, _default)
    setattr(m, 'load_' + k, make_f(paths, default))
    setattr(m, 'redis_load_' + k, make_redis_f(paths, default))


def load_secrets():
    return load_login(), load_password()


def _load(paths, default=None):
    with open('user_config.local', 'r') as f:
        j = json5.loads(f.read())
        return _json_path(j, paths) or default


def _load_from_redis(u, paths, default=None):
    import data
    j = data.get_latest_config(u)
    return _json_path(j, paths) or default


# publish a config file to redis
# this does not work any more since we do key based user config.
# TODO: fix this
def publish(filename):
    import time
    import os.path
    import json
    import data

    with open(filename, 'r') as f:
        j = json5.loads(f.read())
        u = j.get('login')
        # TODO: check integrity of config

        if u is None:
            raise Exception("login should present")
        else:
            j['published_time'] = int(time.time())
            j['modified_time'] = int(os.path.getmtime(filename))
            print("The new config json is:\n")
            print(json.dumps(j, indent=4))

            if data.set_config(u, j):
                print("Upload complete")


if __name__ == '__main__':
    # action = sys.argv[1]
    # if action == 'publish':
    #     filename = sys.argv[2]
    #     publish(filename)
    # elif action == 'get':
    print(load_secrets())
