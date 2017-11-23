import secret_reader
import redis
import json

REDIS_DB = 1


NAMESPACE_JSON = 'user_json:'
NAMESPACE_FOLLOWED = 'followed:'


JSON_TTL = 3600 * 24 * 7  # in seconds


redis_host = secret_reader.load_redis_host()
redis_port = secret_reader.load_redis_port()
_redis = None
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
    _redis.set(NAMESPACE_JSON + str(u), json.dumps(j), JSON_TTL)


def set_followed(u, i):
    _redis.sadd(NAMESPACE_FOLLOWED + str(u), str(i))


def is_followed(u, i):
    return _redis.sismember(NAMESPACE_FOLLOWED + str(u), str(i))


# Below is back fill code to put .txt data into redis
# once ran for every user, below code can be removed
data_filename = 'followed.txt'
my_u = secret_reader.load_secrets()[0]

with open(data_filename, 'r+') as fr:
    for id in fr.readlines():
        id = id.strip()
        if not is_followed(my_u, id):
            print('back filling %s to redis' % (id))
            set_followed(my_u, id)

