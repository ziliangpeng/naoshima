import secret_reader
import redis
import json

REDIS_DB = 1


NAMESPACE_JSON = 'user_json:'


JSON_TTL = 3600 * 24 * 7  # in seconds


redis_host = secret_reader.load_redis_host()
redis_port = secret_reader.load_redis_port()

_redis = None
if redis_host != None and redis_port != None:
    _redis = redis.Redis(redis_host, redis_port, REDIS_DB)



def get_json_by_username(u):
    if _redis == None:
        return None
    else:
        j = _redis.get(NAMESPACE_JSON + str(u))
        if j != None:
            return json.loads(j)
        else:
            return None


def set_json_by_username(u, j):
    if _redis == None:
        return None
    else:
        _redis.set(NAMESPACE_JSON + str(u), json.dumps(j), JSON_TTL)


data_filename = 'followed.txt'
followed = set()

with open(data_filename, 'r+') as fr:
    for id in fr.readlines():
        followed.add(id.strip())

f = open(data_filename, 'a+')


def follow(id):
    followed.add(id)
    f.write(str(id) + '\n')
    f.flush()


def is_followed(id):
    return id in followed
