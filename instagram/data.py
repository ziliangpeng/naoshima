import secret_reader
import redis
import json
import time

REDIS_DB = 1


NAMESPACE_JSON = 'user_json:'
NAMESPACE_FOLLOWED = 'followed:'
NAMESPACE_FOLLOW_DATE = 'follow_date:'
NAMESPACE_FOLLOWED_BACK = 'followed_back:'
NAMESPACE_FOLLOWED_BACK_DATE = 'followed_back_date:'
NAMESPACE_ID_NAME_MAP = 'id_to_name:'


DEFAULT_TTL = 3600 * 24 * 7  # in seconds


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
    _redis.sadd(NAMESPACE_FOLLOWED_BACK + str(u), str(i))

