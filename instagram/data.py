import redis
import json
import time
import pickle

import storage_config_reader

REDIS_DB = 1
REDIS_CACHE_DB = 3


NAMESPACE_JSON = 'user_json:'
NAMESPACE_FOLLOWED = 'followed:'
NAMESPACE_FOLLOW_DATE = 'follow_date:'
NAMESPACE_FOLLOWED_BACK = 'followed_back:'
NAMESPACE_FOLLOWED_BACK_DATE = 'followed_back_date:'
NAMESPACE_ID_NAME_MAP = 'id_to_name:'
NAMESPACE_POSTED = 'posted:'
NAMESPACE_CONFIG = 'config:'
NAMESPACE_USER_TO_FOLLOW = 'user_to_follow:'

KEY_ID_SAVED_SESSIONS_MAP = 'sessions'
KEY_POST_ID_CODE_MAP = 'post_id_to_code'
KEY_POST_ID_TIME_MAP = 'post_id_to_time'


# I increased it from 7 days to 30 days. This hurts the accuracy especially freshness but this is fine
# we are mining user from a huge sea, we may miss some users but greatly improve performance.
# Update(2018-01-23): 7 days should be enough
# Update(2018-04-08): 1 day is more than enough
# Update(2018-04-08): fuck it. 1 hour
DEFAULT_TTL = 3600 * 24 * 1  # in seconds


redis_host = storage_config_reader.load_redis_host()
redis_port = storage_config_reader.load_redis_port()
if redis_host is not None and redis_port is not None:
    _redis = redis.Redis(redis_host, redis_port, REDIS_DB)
else:
    raise Exception("redis config must present!")

redis_cache_host = storage_config_reader.load_redis_cache_host()
redis_cache_port = storage_config_reader.load_redis_cache_port()
if redis_cache_host is not None and redis_cache_port is not None:
    _redis_cache = redis.Redis(redis_cache_host, redis_cache_port, REDIS_CACHE_DB)
else:
    raise Exception("redis-cache config must present!")

def get_latest_config(u: str) -> dict:
    key = NAMESPACE_CONFIG + u
    j = _redis.lindex(key, -1)
    return _load_json_if_exist(j)


def get_all_config(u: str):
    raise NotImplementedError()


def set_config(u: str, config: dict):
    key = NAMESPACE_CONFIG + u
    latest_config = get_latest_config(u)
    if latest_config is not None and latest_config.get('modified_time') == config.get('modified_time'):
        print("The new config match the existing config. Not uploading")
        return False
    else:
        if latest_config is None:
            print("No previous config for", u)
        print("Uploading new config for", u)
        j = json.dumps(config)
        _redis.rpush(key, j)
        return True


def add_user_to_follow(u, to_follow_id, score):
    key = NAMESPACE_USER_TO_FOLLOW + u
    # the redis command have (score, key) format, but python implementation accidentally swapped the 2 args
    # so in python we have to use (key, score)
    _redis.zadd(key, to_follow_id, score)  # This can automatically update the score if exist
    pass


def get_user_to_follow(u):
    key = NAMESPACE_USER_TO_FOLLOW + u
    # To do a ZPOP, we 1) get largest element, 2) remove it from redis
    # This is not atomic, but we have only one consumer for each sorted set so it's fine
    to_fo_id = _redis.zrange(key, -1, -1)  # get the element with largest score
    _redis.zrem(key, to_fo_id)
    return to_fo_id


def get_json_by_username(u):
    j = _redis_cache.get(NAMESPACE_JSON + str(u))
    return _load_json_if_exist(j)


def set_json_by_username(u, j):
    _redis_cache.set(NAMESPACE_JSON + str(u), json.dumps(j), DEFAULT_TTL)


def save_session(name, s):
    print(type(s))
    _redis.hset(KEY_ID_SAVED_SESSIONS_MAP, name, pickle.dumps(s))


def get_session(name):
    pickled_session = _redis.hget(KEY_ID_SAVED_SESSIONS_MAP, name)
    if pickled_session:
        try:
            return pickle.loads(pickled_session)
        except Exception as e:
            print('Exception in unpickling a session object:', e)
    return None


def set_id_to_name(i, u):
    _redis.set(NAMESPACE_ID_NAME_MAP + str(i), str(u))


def get_id_to_name(i):
    ret = _redis.get(NAMESPACE_ID_NAME_MAP + str(i))
    if not ret:
        raise BaseException("id-name mapping for %s should present" % (i))
    return ret.decode()  # byte array to string


def set_followed(u, i):
    _redis.hset(NAMESPACE_FOLLOW_DATE + str(u), str(i), int(time.time()))
    _redis.sadd(NAMESPACE_FOLLOWED + str(u), str(i))


def is_followed(u, i):
    return _redis.sismember(NAMESPACE_FOLLOWED + str(u), str(i))


def set_followed_back(u, i):
    # TODO: when everyone backfilled followed-back data (by running score.py for everyone), start to record date
    _redis.sadd(NAMESPACE_FOLLOWED_BACK + str(u), str(i))


def set_posted(u, i, code):
    _redis.hset(KEY_POST_ID_CODE_MAP, str(i), str(code))
    _redis.hset(KEY_POST_ID_TIME_MAP, str(i), int(time.time()))
    _redis.sadd(NAMESPACE_POSTED + str(u), str(i))


def is_posted(u, i):
    return _redis.sismember(NAMESPACE_POSTED + str(u), str(i))


def _load_json_if_exist(j):
    if j is not None:
        return json.loads(j)
    else:
        return None
