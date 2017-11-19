import json


def load_secrets():
    return _load(["login"]), _load(["password"])


def load_user_id():
    return _load(["id"])


def load_whitelist():
    return _load(["whitelist"], [])


def load_conditions():
    return _load(["conditions"], {})


def load_like_per_fo():
    return _load(["like_per_fo"], 1)


def load_comment_pool():
    return _load(["comments"], [u"これは素晴らしい写真です", u"私はこの写真が好き"])


def load_redis_host():
    return _load(["redis", "host"])


def load_redis_port():
    return _load(["redis", "port"])


def _load(keys, default=None):
    with open('secret.local', 'r') as f:
        j = json.loads(f.read())
        for k in keys:
            if k in j:
                j = j[k]
            else:
                return default
        return j
