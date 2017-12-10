import secret_reader
import redis
import json
import time
import pickle

REDIS_DB = 1


NAMESPACE_JSON = 'user_json:'
NAMESPACE_FOLLOWED = 'followed:'
NAMESPACE_FOLLOW_DATE = 'follow_date:'
NAMESPACE_FOLLOWED_BACK = 'followed_back:'
NAMESPACE_FOLLOWED_BACK_DATE = 'followed_back_date:'
NAMESPACE_ID_NAME_MAP = 'id_to_name:'
NAMESPACE_POSTED = 'posted:'

KEY_ID_SAVED_SESSIONS_MAP = 'sessions'
KEY_POST_ID_CODE_MAP = 'post_id_to_code'
KEY_POST_ID_TIME_MAP = 'post_id_to_time'


# I increased it from 7 days to 30 days. This hurts the accuracy especially freshness but this is fine
# we are mining user from a huge sea, we may miss some users but greatly improve performance.
DEFAULT_TTL = 3600 * 24 * 30  # in seconds


redis_host = secret_reader.load_redis_host()
redis_port = secret_reader.load_redis_port()
if redis_host != None and redis_port != None:
    _redis = redis.Redis(redis_host, redis_port, REDIS_DB)
else:
    raise Exception("redis config must present!")


def get_json_by_username(u):
    j = _redis.get(NAMESPACE_JSON + str(u))
    if j != None:
        return json.loads(j)
    else:
        return None


def set_json_by_username(u, j):
    _redis.set(NAMESPACE_JSON + str(u), json.dumps(j), DEFAULT_TTL)


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
    return ret.decode()  #byte array to string


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
